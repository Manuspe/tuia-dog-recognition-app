from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models

from lib.config import settings
from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.models.cnn_custom import CNNCustom
from lib.preprocessing import (
    build_train_augmenter,
    l2_normalize_vector,
    preprocess_bgr_batch,
    preprocess_bgr_numpy,
)
from lib.splits import DogBreedDataset, load_splits

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _extract_torch_features(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "extract_features"):
            return model.extract_features(tensor)
        if isinstance(model, models.ResNet):
            x = model.conv1(tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            return torch.flatten(x, 1)
        raise ValueError(
            f"No se pudo extraer embedding de la arquitectura '{type(model).__name__}'."
        )

    def _build_model_for_training(
        self, num_classes: int, classes: list[str]
    ) -> tuple[nn.Module, list[dict[str, Any]]]:
        lr = float(settings.learning_rate)
        if self.active_model_name == "resnet18_finetuned":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            model.classes = list(classes)  # type: ignore[attr-defined]
            model.image_size = self.image_size  # type: ignore[attr-defined]
            model.embedding_dim = in_features  # type: ignore[attr-defined]

            strategy = settings.resnet_finetune_strategy.strip().lower()
            if strategy == "head_only":
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc.parameters():
                    param.requires_grad = True
                param_groups = [{"params": list(model.fc.parameters()), "lr": lr}]
            elif strategy == "partial":
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.layer4.parameters():
                    param.requires_grad = True
                for param in model.fc.parameters():
                    param.requires_grad = True
                param_groups = [
                    {"params": list(model.layer4.parameters()), "lr": lr * 0.1},
                    {"params": list(model.fc.parameters()), "lr": lr},
                ]
            elif strategy == "full":
                for param in model.parameters():
                    param.requires_grad = True
                backbone_params = [
                    p for name, p in model.named_parameters() if not name.startswith("fc.")
                ]
                param_groups = [
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": list(model.fc.parameters()), "lr": lr},
                ]
            else:
                raise ValueError(
                    f"RESNET_FINETUNE_STRATEGY invalida: '{strategy}'. "
                    "Opciones validas: head_only, partial, full."
                )
            return model, param_groups

        if self.active_model_name == "cnn_custom":
            model = CNNCustom(
                num_classes=num_classes,
                embedding_dim=settings.embedding_dim,
                dropout=settings.cnn_dropout,
                classes=classes,
            )
            param_groups = [{"params": list(model.parameters()), "lr": lr}]
            return model, param_groups

        raise ValueError(f"Modelo no soportado para entrenamiento: '{self.active_model_name}'")

    @staticmethod
    def _compute_class_weights(
        train_samples: list[tuple[Path, int]], num_classes: int
    ) -> torch.Tensor:
        counts = np.bincount([label for _, label in train_samples], minlength=num_classes).astype(
            np.float32
        )
        counts = np.maximum(counts, 1.0)
        weights = len(train_samples) / (num_classes * counts)
        weights = weights / np.mean(weights)
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _resolve_model_classes(model: Any, fallback_classes: list[str]) -> list[str]:
        if isinstance(model, onnxruntime.InferenceSession):
            raw = model.get_modelmeta().custom_metadata_map.get("classes")
            if raw:
                return list(json.loads(raw))
            return fallback_classes
        model_classes = getattr(model, "classes", None)
        if model_classes:
            return list(model_classes)
        return fallback_classes

    def train_classifier(self) -> None:
        """
        Entrena el modelo activo (resnet18_finetuned o cnn_custom) usando los
        splits de 1-pre-processing.ipynb, preprocesamiento compartido, data
        augmentation on-the-fly sobre train y early stopping sobre valid_loss.
        Guarda el modelo completo (con atributo .classes) en self.active_checkpoint
        y el historial por epoca en self.output_path.
        """
        self._set_seed(settings.random_seed)
        splits, classes = load_splits(self.dataset_path, settings.splits_path)
        train_samples = splits.get("train", [])
        valid_samples = splits.get("valid", [])
        if not train_samples or not valid_samples:
            raise ValueError(
                "Los splits 'train' y 'valid' no pueden estar vacios para entrenar."
            )

        num_classes = len(classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        augmenter = build_train_augmenter(seed=settings.random_seed)

        train_ds = DogBreedDataset(train_samples, image_size=self.image_size, augmenter=augmenter)
        valid_ds = DogBreedDataset(valid_samples, image_size=self.image_size, augmenter=None)

        num_workers = 0 if os.name == "nt" else min(2, settings.max_workers)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model, param_groups = self._build_model_for_training(num_classes, classes)
        model.to(device)

        class_weights = (
            self._compute_class_weights(train_samples, num_classes).to(device)
            if settings.use_class_weights
            else None
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=settings.label_smoothing,
        )
        optimizer = optim.AdamW(param_groups, weight_decay=settings.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        history: dict[str, Any] = {
            "model": self.active_model_name,
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
            "lr": [],
            "best_epoch": 0,
            "best_valid_loss": float("inf"),
            "classes": classes,
            "config": {
                "seed": settings.random_seed,
                "batch_size": settings.batch_size,
                "epochs": settings.train_epochs,
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "label_smoothing": settings.label_smoothing,
                "early_stopping_patience": settings.early_stopping_patience,
                "resnet_finetune_strategy": settings.resnet_finetune_strategy,
                "use_class_weights": settings.use_class_weights,
                "image_size": self.image_size,
                "embedding_dim": settings.embedding_dim,
            },
        }

        best_valid_loss = float("inf")
        epochs_no_improve = 0
        logger.info(
            "Iniciando entrenamiento '%s' (%d clases, train=%d, valid=%d, device=%s)",
            self.active_model_name,
            num_classes,
            len(train_samples),
            len(valid_samples),
            device,
        )

        for epoch in range(1, settings.train_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                bs = inputs.size(0)
                train_loss_sum += float(loss.item()) * bs
                train_correct += int((preds == targets).sum().item())
                train_total += bs

            model.eval()
            valid_loss_sum = 0.0
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    preds = outputs.argmax(dim=1)
                    bs = inputs.size(0)
                    valid_loss_sum += float(loss.item()) * bs
                    valid_correct += int((preds == targets).sum().item())
                    valid_total += bs

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)
            valid_loss = valid_loss_sum / max(valid_total, 1)
            valid_acc = valid_correct / max(valid_total, 1)
            current_lr = float(optimizer.param_groups[-1]["lr"])

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["valid_loss"].append(valid_loss)
            history["valid_acc"].append(valid_acc)
            history["lr"].append(current_lr)

            scheduler.step(valid_loss)

            logger.info(
                "Epoca %d/%d - train_loss=%.4f train_acc=%.4f | valid_loss=%.4f valid_acc=%.4f | lr=%.2e",
                epoch,
                settings.train_epochs,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                current_lr,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_no_improve = 0
                history["best_epoch"] = epoch
                history["best_valid_loss"] = valid_loss

                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                model_to_save = copy.deepcopy(model).cpu().eval()
                torch.save(model_to_save, self.active_checkpoint)
                logger.info(
                    "Mejor checkpoint guardado en %s (epoca %d, valid_loss=%.4f)",
                    self.active_checkpoint,
                    epoch,
                    valid_loss,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= settings.early_stopping_patience:
                    logger.info(
                        "Early stopping activado en epoca %d tras %d epocas sin mejora.",
                        epoch,
                        epochs_no_improve,
                    )
                    break

        self._loaded.pop(self.active_model_name, None)
        self.output_path.mkdir(parents=True, exist_ok=True)
        history_path = self.output_path / f"{self.active_model_name}_history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")
        logger.info("Historial de entrenamiento guardado en %s", history_path)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el split de test.
        Retorna {"accuracy", "precision", "recall", "specificity", "f1"} (macro)
        y guarda el detalle de evaluacion en self.output_path para analisis.
        """
        splits, split_classes = load_splits(self.dataset_path, settings.splits_path)
        test_samples = splits.get("test", [])
        if not test_samples:
            raise ValueError("El split 'test' esta vacio; no se puede evaluar el clasificador.")

        model = self.load_model()
        classes = self._resolve_model_classes(model, split_classes)
        model_class_to_idx = {name: idx for idx, name in enumerate(classes)}
        if classes != split_classes:
            remapped: list[tuple[Path, int]] = []
            for path, old_idx in test_samples:
                breed_name = split_classes[old_idx]
                if breed_name in model_class_to_idx:
                    remapped.append((path, model_class_to_idx[breed_name]))
            test_samples = remapped

        test_ds = DogBreedDataset(
            test_samples, image_size=self.image_size, augmenter=None, return_path=True
        )
        num_workers = 0 if os.name == "nt" else min(2, settings.max_workers)
        test_loader = DataLoader(
            test_ds, batch_size=settings.batch_size, shuffle=False, num_workers=num_workers
        )

        y_true: list[int] = []
        y_pred: list[int] = []
        y_probs: list[list[float]] = []
        paths: list[str] = []

        if isinstance(model, onnxruntime.InferenceSession):
            input_name = model.get_inputs()[0].name
            for inputs, targets, batch_paths in test_loader:
                logits_np = np.asarray(
                    model.run(None, {input_name: inputs.numpy().astype(np.float32)})[0],
                    dtype=np.float32,
                )
                shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
                exp_vals = np.exp(shifted)
                probs_np = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
                preds_np = np.argmax(probs_np, axis=1)

                y_true.extend(int(v) for v in targets.numpy())
                y_pred.extend(int(v) for v in preds_np)
                y_probs.extend([float(p) for p in row] for row in probs_np)
                paths.extend(str(p) for p in batch_paths)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                for inputs, targets, batch_paths in test_loader:
                    logits = model(inputs.to(device))
                    probs = torch.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    y_true.extend(int(v) for v in targets.numpy())
                    y_pred.extend(int(v) for v in preds.cpu().numpy())
                    y_probs.extend([float(p) for p in row] for row in probs.cpu().numpy())
                    paths.extend(str(p) for p in batch_paths)

        num_classes = len(classes)
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

        total = int(cm.sum())
        acc = float(np.trace(cm) / total) if total > 0 else 0.0

        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        specificities: list[float] = []
        per_class: list[dict[str, Any]] = []

        for idx, breed in enumerate(classes):
            tp = int(cm[idx, idx])
            fp = int(cm[:, idx].sum() - tp)
            fn = int(cm[idx, :].sum() - tp)
            tn = int(total - tp - fp - fn)
            prec_i, rec_i, f1_i = precision_recall_f1(tp, fp, fn)
            spec_i = specificity(tn, fp)
            precisions.append(prec_i)
            recalls.append(rec_i)
            f1s.append(f1_i)
            specificities.append(spec_i)
            per_class.append(
                {
                    "breed": breed,
                    "support": int(cm[idx, :].sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(prec_i, 6),
                    "recall": round(rec_i, 6),
                    "specificity": round(spec_i, 6),
                    "f1": round(f1_i, 6),
                }
            )

        metrics = {
            "accuracy": round(acc, 6),
            "precision": round(float(np.mean(precisions)), 6),
            "recall": round(float(np.mean(recalls)), 6),
            "specificity": round(float(np.mean(specificities)), 6),
            "f1": round(float(np.mean(f1s)), 6),
        }

        self.output_path.mkdir(parents=True, exist_ok=True)
        eval_path = self.output_path / f"{self.active_model_name}_eval.json"
        eval_payload = {
            "model": self.active_model_name,
            "metrics": metrics,
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "paths": paths,
        }
        eval_path.write_text(
            json.dumps(eval_payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        logger.info("Evaluacion '%s' completada: %s (detalle en %s)", self.active_model_name, metrics, eval_path)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Extrae el embedding L2-normalizado de la capa previa al clasificador del
        modelo activo (.pth o .onnx con salida 'embedding') usando el mismo
        preprocesamiento que en entrenamiento.
        """
        model = self.load_model()
        if isinstance(model, onnxruntime.InferenceSession):
            output_names = [out.name for out in model.get_outputs()]
            if "embedding" not in output_names:
                raise ValueError(
                    "El modelo ONNX activo no expone una salida 'embedding'; "
                    "usa el checkpoint .pth o exporta ONNX con salidas ['logits', 'embedding']."
                )
            inp = preprocess_bgr_numpy(image, image_size=self.image_size)
            input_name = model.get_inputs()[0].name
            outputs = model.run(["embedding"], {input_name: inp})
            return l2_normalize_vector(outputs[0])

        device = next(model.parameters()).device
        tensor = preprocess_bgr_batch(image, image_size=self.image_size).to(device)
        model.eval()
        with torch.no_grad():
            emb = self._extract_torch_features(model, tensor)
        return l2_normalize_vector(emb)
