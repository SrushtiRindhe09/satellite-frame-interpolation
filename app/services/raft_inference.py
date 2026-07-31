import torch

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights
)


def load_raft_model():
    """
    Load the RAFT optical flow model with pretrained weights.
    Original logic from models/raft_model.py.
    """
    print("Loading RAFT model...")

    weights = Raft_Large_Weights.DEFAULT

    model = raft_large(weights=weights)

    model.eval()

    print("[OK] RAFT Model Loaded Successfully")

    return model


def predict_optical_flow(model, image1, image2):
    """
    Run RAFT inference to predict optical flow between two image tensors.
    Original logic from models/raft_model.py.
    """
    with torch.no_grad():

        flow_predictions = model(image1, image2)

    return flow_predictions
