"""Preprocesamiento y data augmentation compartidos entre notebooks y servicios.

Garantiza paridad exacta entre entrenamiento, evaluacion, extraccion de embeddings
y clasificacion en produccion (orden BGR->RGB, letterbox preservando aspect ratio
y normalizacion con estadisticas de ImageNet).
"""
from __future__ import annotations

import inspect
from typing import Any

import cv2
import numpy as np
import torch

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
# Valor uint8 cercano a la media de ImageNet para que los bordes de padding
# queden proximos a 0.0 tras normalizar y no introduzcan bordes artificiales.
LETTERBOX_PAD_RGB: tuple[int, int, int] = (124, 116, 104)

_MEAN_ARR = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_STD_ARR = np.asarray(IMAGENET_STD, dtype=np.float32)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen OpenCV (BGR/GRAY/BGRA uint8) a RGB uint8."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("La imagen de entrada esta vacia o no es un np.ndarray valido.")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    if image_bgr.ndim != 3:
        raise ValueError(f"Dimension de imagen no soportada: {image_bgr.shape}")
    channels = image_bgr.shape[2]
    if channels == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    if channels == 1:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Cantidad de canales no soportada: {channels}")


def letterbox_rgb(
    image_rgb: np.ndarray,
    target_size: int = 224,
    pad_color: tuple[int, int, int] = LETTERBOX_PAD_RGB,
) -> np.ndarray:
    """Redimensiona preservando la relacion de aspecto y aplica padding cuadrado."""
    if target_size <= 0:
        raise ValueError(f"target_size debe ser positivo, recibido: {target_size}")
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Imagen vacia con shape {image_rgb.shape}")

    scale = min(target_size / h, target_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )


def normalize_rgb_to_chw(image_rgb: np.ndarray) -> np.ndarray:
    """Normaliza imagen RGB uint8 [0..255] con media/desvio de ImageNet a float32 CHW."""
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - _MEAN_ARR) / _STD_ARR
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def preprocess_rgb(image_rgb: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Preprocesa una imagen RGB uint8 a tensor float32 (3, image_size, image_size)."""
    boxed = letterbox_rgb(image_rgb, target_size=image_size)
    chw = normalize_rgb_to_chw(boxed)
    return torch.from_numpy(np.ascontiguousarray(chw))


def preprocess_bgr(image_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Preprocesa una imagen BGR uint8 a tensor float32 (3, image_size, image_size)."""
    return preprocess_rgb(bgr_to_rgb(image_bgr), image_size=image_size)


def preprocess_bgr_batch(image_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Preprocesa una imagen BGR uint8 a tensor (1, 3, image_size, image_size)."""
    return preprocess_bgr(image_bgr, image_size=image_size).unsqueeze(0)


def preprocess_bgr_numpy(image_bgr: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Preprocesa una imagen BGR uint8 a np.ndarray float32 (1, 3, image_size, image_size)."""
    boxed = letterbox_rgb(bgr_to_rgb(image_bgr), target_size=image_size)
    chw = normalize_rgb_to_chw(boxed)
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def l2_normalize_vector(vector: np.ndarray | torch.Tensor) -> list[float]:
    """Aplica normalizacion L2 segura y convierte a list[float] nativa de Python."""
    if isinstance(vector, torch.Tensor):
        arr = vector.detach().cpu().numpy()
    else:
        arr = np.asarray(vector)
    flat = arr.astype(np.float64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm > 0.0:
        flat = flat / norm
    return [float(v) for v in flat]


def _make_gaussian_blur(alb: Any) -> Any:
    sig = inspect.signature(alb.GaussianBlur.__init__)
    if "blur_range" in sig.parameters:
        return alb.GaussianBlur(blur_range=(3, 5), sigma_range=(0.2, 1.2), p=0.25)
    return alb.GaussianBlur(blur_limit=(3, 5), p=0.25)


def _make_gauss_noise(alb: Any) -> Any:
    sig = inspect.signature(alb.GaussNoise.__init__)
    if "std_range" in sig.parameters:
        return alb.GaussNoise(std_range=(0.02, 0.06), mean_range=(0.0, 0.0), p=0.35)
    return alb.GaussNoise(var_limit=(10.0, 40.0), p=0.35)


def build_train_augmenter(seed: int | None = None) -> Any:
    """Pipeline de data augmentation (Albumentations) aplicable solo sobre train.

    Incluye transformaciones realistas para perros sin alterar rasgos de raza:
      - HorizontalFlip (simetria natural)
      - Affine leve (rotacion +/-15 deg, escala +/-10%, traslacion +/-5%)
      - RandomBrightnessContrast (variaciones de iluminacion y exposicion)
      - HueSaturationValue acotado (sin alterar color del pelaje)
      - GaussianBlur (desenfoque por foco o movimiento)
      - GaussNoise (ruido de sensor)
    """
    import albumentations as A

    affine_sig = inspect.signature(A.Affine.__init__)
    affine_kwargs: dict[str, Any] = {
        "scale": (0.9, 1.1),
        "translate_percent": (-0.05, 0.05),
        "rotate": (-15, 15),
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 0.7,
    }
    if "fill" in affine_sig.parameters:
        affine_kwargs["fill"] = LETTERBOX_PAD_RGB
    elif "cval" in affine_sig.parameters:
        affine_kwargs["cval"] = LETTERBOX_PAD_RGB

    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.Affine(**affine_kwargs),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        _make_gaussian_blur(A),
        _make_gauss_noise(A),
    ]
    compose_kwargs: dict[str, Any] = {}
    if seed is not None and "seed" in inspect.signature(A.Compose.__init__).parameters:
        compose_kwargs["seed"] = seed
    return A.Compose(transforms_list, **compose_kwargs)


def preprocess_bgr_train(
    image_bgr: np.ndarray,
    image_size: int = 224,
    augmenter: Any | None = None,
) -> torch.Tensor:
    """Preprocesa BGR uint8 aplicando letterbox y augmentation opcional para train."""
    boxed = letterbox_rgb(bgr_to_rgb(image_bgr), target_size=image_size)
    if augmenter is not None:
        boxed = augmenter(image=boxed)["image"]
    chw = normalize_rgb_to_chw(boxed)
    return torch.from_numpy(np.ascontiguousarray(chw))
