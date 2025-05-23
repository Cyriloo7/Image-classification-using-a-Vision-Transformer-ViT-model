import os
import timm
import torch
import torch.nn as nn


def build_vit(num_classes: int,
              model_out: str,
              model_name: str = 'vit_base_patch16_224',
              pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained Vision Transformer, replace its head,
    and optionally load existing checkpoint from `model_out`.
    """
    # Create the base model
    model = timm.create_model(model_name, pretrained=pretrained)
    # Replace classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    # If a checkpoint exists, load it
    if os.path.exists(model_out):
        state = torch.load(model_out, map_location='cpu')
        # If saved as state_dict or full model
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state
        print(f"[Model] Loaded checkpoint from {model_out}")

    return model



def build_vit_prediction(num_classes: int,
              model_out: str,
              model_name: str = 'vit_base_patch16_224',
              pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained Vision Transformer, replace its head,
    and optionally load existing checkpoint from `model_out`.
    """
    # Create the base model
    model = timm.create_model(model_name, pretrained=pretrained)
    # Replace classification head
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    # If a checkpoint exists, load it
    if os.path.exists(model_out):
        state = torch.load(model_out, map_location='cpu')
        # If saved as state_dict or full model
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state
        print(f"[Model] Loaded checkpoint from {model_out}")

        return model
    
    else:
        print(f"[Model] No checkpoint found at {model_out}")
        raise FileNotFoundError(f"No checkpoint found at {model_out}")