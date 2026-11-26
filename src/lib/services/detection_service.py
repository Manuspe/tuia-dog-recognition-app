from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime
import torch

from lib.config import settings
from lib.preprocessing import preprocess_bgr_batch, preprocess_bgr_numpy
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService
from lib.splits import normalize_breed_name

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def _get_yolo(self) -> Any:
        cached = getattr(self, "_yolo", None)
        if cached is not None:
            return cached

        from ultralytics import YOLO

        model_path = Path(self.yolo_model_name)
        if model_path.parent != Path("."):
            model_path.parent.mkdir(parents=True, exist_ok=True)
        self._yolo = YOLO(self.yolo_model_name)
        logger.info("Detector YOLO cargado: %s", self.yolo_model_name)
        return self._yolo

    def _resolve_classes(self, model: Any) -> list[str]:
        if isinstance(model, onnxruntime.InferenceSession):
            raw = model.get_modelmeta().custom_metadata_map.get("classes")
            if raw:
                return list(json.loads(raw))
        else:
            model_classes = getattr(model, "classes", None)
            if model_classes:
                return list(model_classes)

        train_dir = self.classifier.dataset_path / "train"
        if train_dir.is_dir():
            classes = sorted(
                normalize_breed_name(p.name) for p in train_dir.iterdir() if p.is_dir()
            )
            if classes:
                return classes

        raise ValueError(
            "El checkpoint cargado no contiene la lista de razas (.classes / metadata) "
            f"y no se encontro {train_dir}."
        )

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en una imagen BGR uint8 usando el
        modelo YOLO preentrenado en COCO configurado en .env.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles enteros.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        yolo = self._get_yolo()
        results = yolo(
            image,
            conf=self.conf_threshold,
            classes=[self.dog_class_id],
            verbose=False,
        )

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        if not results:
            return detections

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id != self.dog_class_id or conf < self.conf_threshold:
                continue
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (int(round(float(v))) for v in coords[:4])
            detections.append(((x1, y1, x2, y2), conf))

        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza de un recorte BGR uint8 usando el modelo activo de
        Etapa 2 (.pth o .onnx) con el preprocesamiento compartido.
        """
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return ("unknown", 0.0)

        model = self.classifier.load_model()
        classes = self._resolve_classes(model)

        if isinstance(model, onnxruntime.InferenceSession):
            inp = preprocess_bgr_numpy(crop, image_size=self.classifier.image_size)
            input_name = model.get_inputs()[0].name
            logits = np.asarray(model.run(None, {input_name: inp})[0], dtype=np.float32).reshape(-1)
            shifted = logits - np.max(logits)
            exp_vals = np.exp(shifted)
            probs = exp_vals / np.sum(exp_vals)
            idx = int(np.argmax(probs))
            score = float(probs[idx])
        else:
            device = next(model.parameters()).device
            inp = preprocess_bgr_batch(crop, image_size=self.classifier.image_size).to(device)
            model.eval()
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)
                conf_t, idx_t = torch.max(probs, dim=1)
            idx = int(idx_t.item())
            score = float(conf_t.item())

        if settings.classifier_conf_threshold > 0.0 and score < settings.classifier_conf_threshold:
            return ("unknown", score)

        breed = classes[idx] if 0 <= idx < len(classes) else "unknown"
        return (breed, score)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        # El modelo elegido vale solo para esta clasificacion: se restaura el activo para
        # que /detect siga usando el modelo configurado.
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
