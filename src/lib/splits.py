"""Utilidades reproducibles de splits y Dataset PyTorch para 70 Dog Breeds."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from lib.preprocessing import preprocess_bgr, preprocess_bgr_train

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class SplitSample:
    """Una muestra indexada en un split del dataset."""

    rel_path: str
    breed: str
    split: str


def normalize_breed_name(name: str) -> str:
    """Normaliza espacios multiples en nombres de razas (ej. 'American  Spaniel')."""
    return " ".join(name.strip().split())


def scan_dataset_splits(
    dataset_path: Path,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> list[SplitSample]:
    """Escanea las subcarpetas de splits del dataset y normaliza los nombres de raza."""
    samples: list[SplitSample] = []
    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.is_dir():
            continue
        for breed_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            breed = normalize_breed_name(breed_dir.name)
            for img_path in sorted(breed_dir.iterdir(), key=lambda p: p.name):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_SUFFIXES:
                    rel = img_path.relative_to(dataset_path).as_posix()
                    samples.append(SplitSample(rel_path=rel, breed=breed, split=split))
    return samples


def save_splits(samples: list[SplitSample], csv_path: Path) -> None:
    """Guarda los splits finales en CSV UTF-8 para reutilizar en notebooks y scripts."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rel_path", "breed", "split"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "rel_path": sample.rel_path,
                    "breed": normalize_breed_name(sample.breed),
                    "split": sample.split,
                }
            )
    logger.info("Splits guardados en %s (%d muestras)", csv_path, len(samples))


def load_splits_csv(csv_path: Path) -> list[SplitSample]:
    """Lee un archivo CSV de splits generado por 1-pre-processing.ipynb."""
    samples: list[SplitSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(
                SplitSample(
                    rel_path=row["rel_path"].strip(),
                    breed=normalize_breed_name(row["breed"]),
                    split=row["split"].strip(),
                )
            )
    return samples


def load_splits(
    dataset_path: Path,
    splits_path: Path | None = None,
) -> tuple[dict[str, list[tuple[Path, int]]], list[str]]:
    """Carga los splits (desde CSV si existe, o escaneando dataset_path) y el orden de clases."""
    if splits_path is not None and splits_path.is_file():
        logger.info("Cargando splits reproducibles desde %s", splits_path)
        raw_samples = load_splits_csv(splits_path)
    else:
        logger.info("No se encontro %s; escaneando carpetas en %s", splits_path, dataset_path)
        raw_samples = scan_dataset_splits(dataset_path)

    if not raw_samples:
        raise FileNotFoundError(
            f"No se encontraron imagenes en {dataset_path}. "
            "Descarga el dataset con scripts/download_dataset.py."
        )

    train_breeds = {s.breed for s in raw_samples if s.split == "train"}
    classes = sorted(train_breeds if train_breeds else {s.breed for s in raw_samples})
    class_to_idx = {breed: idx for idx, breed in enumerate(classes)}

    grouped: dict[str, list[tuple[Path, int]]] = {"train": [], "valid": [], "test": []}
    missing_files = 0
    for sample in raw_samples:
        if sample.breed not in class_to_idx:
            logger.warning("Raza '%s' fuera del diccionario de train; omitida.", sample.breed)
            continue
        full_path = dataset_path / sample.rel_path
        if not full_path.is_file():
            missing_files += 1
            continue
        grouped.setdefault(sample.split, []).append((full_path, class_to_idx[sample.breed]))

    if missing_files:
        logger.warning("%d rutas del CSV de splits no existen en %s.", missing_files, dataset_path)

    return grouped, classes


class DogBreedDataset(Dataset):
    """Dataset PyTorch que lee con OpenCV (BGR) y aplica preprocessing compartido."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_size: int = 224,
        augmenter: Any | None = None,
        return_path: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augmenter = augmenter
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        img_path, label = self.samples[index]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"No se pudo leer la imagen del dataset: {img_path}")

        if self.augmenter is not None:
            tensor = preprocess_bgr_train(image_bgr, image_size=self.image_size, augmenter=self.augmenter)
        else:
            tensor = preprocess_bgr(image_bgr, image_size=self.image_size)

        if self.return_path:
            return tensor, label, str(img_path)
        return tensor, label
