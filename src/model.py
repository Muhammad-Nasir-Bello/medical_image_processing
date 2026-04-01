# src/model.py
import torch.nn as nn
import torchvision.models as models

from src.config import CFG


def build_model():
    """
    ResNet18 with ImageNet weights, last FC swapped to 4-class head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feats = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, CFG.num_classes),
    )

    return model