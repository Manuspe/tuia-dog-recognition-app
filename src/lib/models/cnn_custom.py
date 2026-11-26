"""Arquitecturas CNN propias (Modelo B) ubicadas en modulo importable para torch.load."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bloque residual basico con dos convoluciones 3x3, BatchNorm y atajo."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CNNCustom(nn.Module):
    """CNN propia residual con extraccion explicita de embeddings de dimension configurable."""

    def __init__(
        self,
        num_classes: int = 70,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        classes: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.classes: list[str] = list(classes or [])

        self.conv_stem = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_stem = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, 64, stride=1, num_blocks=2)
        self.layer2 = self._make_stage(64, 128, stride=2, num_blocks=2)
        self.layer3 = self._make_stage(128, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_stage(256, 512, stride=2, num_blocks=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if embedding_dim == 512:
            self.embedding_proj: nn.Module = nn.Identity()
        else:
            self.embedding_proj = nn.Sequential(
                nn.Linear(512, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, stride: int, num_blocks: int
    ) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el vector de embedding de shape (B, embedding_dim) previo al clasificador."""
        x = F.relu(self.bn_stem(self.conv_stem(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.gap(x), 1)
        return self.embedding_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        return self.fc(self.dropout(emb))


class CNNCustomSimple(nn.Module):
    """Variante secuencial sin bloques residuales para experimentos comparativos del Modelo B."""

    def __init__(
        self,
        num_classes: int = 70,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        classes: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.classes: list[str] = list(classes or [])

        def _block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
            _block(256, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        if embedding_dim == 512:
            self.embedding_proj: nn.Module = nn.Identity()
        else:
            self.embedding_proj = nn.Sequential(
                nn.Linear(512, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
            )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el embedding de shape (B, embedding_dim) previo al clasificador."""
        x = self.features(x)
        x = torch.flatten(self.gap(x), 1)
        return self.embedding_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_features(x)
        return self.fc(self.dropout(emb))
