import torch

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights
)

def load_raft_model():

    print("Loading RAFT model...")

    weights = Raft_Large_Weights.DEFAULT

    model = raft_large(weights=weights)

    model.eval()

    print("✅ RAFT Model Loaded Successfully")

    return model


def predict_optical_flow(model, image1, image2):

    with torch.no_grad():

        flow_predictions = model(image1, image2)

    return flow_predictions