import os
import torch

# -------------------------------------------------------------------
# Device Configuration
# -------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RIFE_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "rife")

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
OPTICAL_FLOW_DIR = os.path.join(OUTPUTS_DIR, "optical_flow")
INTERPOLATED_DIR = os.path.join(OUTPUTS_DIR, "interpolated")

# -------------------------------------------------------------------
# Image Processing Configuration
# -------------------------------------------------------------------
TARGET_SIZE = (512, 512)

# -------------------------------------------------------------------
# Create directories on import
# -------------------------------------------------------------------
for directory in [UPLOADS_DIR, OPTICAL_FLOW_DIR, INTERPOLATED_DIR]:
    os.makedirs(directory, exist_ok=True)
