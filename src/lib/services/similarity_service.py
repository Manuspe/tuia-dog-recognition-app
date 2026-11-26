from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from lib.config import settings
from lib.preprocessing import l2_normalize_vector, preprocess_bgr_batch
from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el Pull Request.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver
        # DATA_PATH: las rutas de la galeria se guardan relativas a esta carpeta para que
        # la base indexada desde el entorno local sirva tambien en Docker (y viceversa).
        self.data_root = data_root

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def _get_baseline_backbone_name(self) -> str:
        supported = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")
        candidate = (self.model_name or "").strip().lower()
        if candidate in supported:
            return candidate
        return (getattr(self, "_baseline_model_name", None) or settings.baseline_model_name).strip().lower()

    def _get_backbone(self) -> tuple[nn.Module, torch.device]:
        backbone_name = self._get_baseline_backbone_name()
        cached = getattr(self, "_backbone", None)
        cached_name = getattr(self, "_backbone_name", None)
        if cached is not None and cached_name == backbone_name:
            return cached, self._device

        if backbone_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Identity()
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier = nn.Identity()
        elif backbone_name == "convnext_tiny":
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            model.classifier[2] = nn.Identity()
        else:
            raise ValueError(
                f"Backbone baseline no soportado: '{backbone_name}'. "
                "Opciones validas: resnet18, resnet50, efficientnet_b0, convnext_tiny."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
        self._backbone = model
        self._backbone_name = backbone_name
        self._device = device
        logger.info("Backbone baseline cargado (%s en %s)", backbone_name, device)
        return model, device

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding normalizado L2 de una imagen BGR uint8 usando el
        backbone preentrenado en ImageNet configurado sin su cabeza de clasificacion.
        """
        tensor = preprocess_bgr_batch(image, image_size=self.image_size)
        model, device = self._get_backbone()
        model.eval()
        with torch.no_grad():
            emb = model(tensor.to(device))
        vec = l2_normalize_vector(emb)
        if len(vec) != settings.embedding_dim:
            logger.warning(
                "Dimension de embedding (%d) difiere de EMBEDDING_DIM (%d).",
                len(vec),
                settings.embedding_dim,
            )
        return vec

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares ordenadas
        por score descendente, soportando pgvector y almacenamiento JSON tanto
        para metrica 'cosine' como 'l2'. No completa Neighbor.url.
        """
        if top_k <= 0:
            return []

        metric = self.similarity_metric.lower()
        search_fn = getattr(self.store, "search", None)
        if metric == "cosine" and callable(search_fn):
            records = search_fn(embedding, top_k)
        else:
            records = self.store.all()

        neighbors = [
            Neighbor(
                path=rec.path,
                breed=rec.breed,
                score=round(float(self.similarity(embedding, rec.embedding)), 6),
            )
            for rec in records
        ]
        neighbors.sort(key=lambda item: item.score, reverse=True)
        return neighbors[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza mediante voto mayoritario ponderado por score entre los
        vecinos que superan self.similarity_threshold.

        Si no hay vecinos o el mejor score queda por debajo del umbral, retorna
        ("unknown", best_score). El score retornado corresponde a la similitud
        media de los vecinos de la raza ganadora.
        """
        if not results:
            return ("unknown", 0.0)

        best_score = max(float(item.score) for item in results)
        if best_score < self.similarity_threshold:
            return ("unknown", best_score)

        eligible = [item for item in results if float(item.score) >= self.similarity_threshold]
        weighted_votes: dict[str, float] = {}
        breed_scores: dict[str, list[float]] = {}
        for item in eligible:
            score = float(item.score)
            weighted_votes[item.breed] = weighted_votes.get(item.breed, 0.0) + max(score, 0.0)
            breed_scores.setdefault(item.breed, []).append(score)

        winning_breed = max(
            weighted_votes,
            key=lambda breed: (weighted_votes[breed], max(breed_scores[breed])),
        )
        mean_score = float(np.mean(breed_scores[winning_breed]))
        return (winning_breed, mean_score)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def _to_stored_path(self, image_path: str) -> str:
        """Ruta relativa a DATA_PATH si la imagen esta dentro; si no, la ruta tal cual."""
        if self.data_root is None:
            return str(image_path)
        try:
            return Path(image_path).resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return str(image_path)

    def resolve_path(self, stored_path: str) -> Path:
        """Inversa de _to_stored_path: ruta local de una imagen guardada en la base."""
        path = Path(stored_path)
        if self.data_root is None or path.is_absolute():
            return path
        candidate = self.data_root / path
        if not candidate.exists() and path.exists():
            # Base indexada antes de guardar rutas relativas a DATA_PATH.
            return path
        return candidate

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial.

        La ruta se guarda relativa a DATA_PATH (ej: dataset/train/<raza>/<imagen>.jpg).
        Para leer la imagen a partir de un registro, usar resolve_path(record.path).
        """
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=self._to_stored_path(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(self.resolve_path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
