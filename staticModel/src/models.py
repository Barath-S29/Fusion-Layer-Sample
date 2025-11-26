# src/models.py
import torch.nn as nn
import timm
from torchvision import models

def get_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def get_efficientnet_b0(num_classes, pretrained=True):
    model = timm.create_model('efficientnet_b0', pretrained=pretrained)
    n_feats = model.get_classifier().in_features
    model.classifier = nn.Linear(n_feats, num_classes)
    return model

def build_model(model_name, num_classes, pretrained=False):
    """
    Build a model by name.
    
    Args:
        model_name: 'resnet50' or 'efficientnet'
        num_classes: number of output classes
        pretrained: whether to use pretrained weights
    
    Returns:
        The model architecture
    """
    if model_name == 'resnet50':
        return get_resnet50(num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        return get_efficientnet_b0(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'resnet50' or 'efficientnet'")