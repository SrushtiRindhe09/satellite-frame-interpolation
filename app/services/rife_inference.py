import cv2
import torch
import numpy as np
import warnings
from torch.nn import functional as F

from app.config import DEVICE, RIFE_WEIGHTS_DIR, TARGET_SIZE

warnings.filterwarnings("ignore")


def load_rife_model():
    """
    Load the RIFE HDv3 model with pretrained weights.
    Original logic from ECCV2022-RIFE/inference_img.py.
    """
    print("Loading RIFE HDv3 model...")

    from rife.RIFE_HDv3 import Model

    model = Model()
    model.load_model(RIFE_WEIGHTS_DIR, -1)
    model.eval()
    model.device()

    print("[OK] RIFE HDv3 Model Loaded Successfully")

    return model


def generate_intermediate_frame(model, image1_np: np.ndarray, image2_np: np.ndarray) -> np.ndarray:
    """
    Generate an intermediate frame between two images using RIFE.
    Original logic from ECCV2022-RIFE/inference_img.py.

    Args:
        model: Loaded RIFE model
        image1_np: First image as numpy array (BGR, preprocessed to TARGET_SIZE)
        image2_np: Second image as numpy array (BGR, preprocessed to TARGET_SIZE)

    Returns:
        Interpolated frame as numpy array (BGR, uint8)
    """
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Resize to target size
    img0 = cv2.resize(image1_np, TARGET_SIZE)
    img1 = cv2.resize(image2_np, TARGET_SIZE)

    # Convert to contiguous float tensor and normalize to [0, 1]
    tensor0 = torch.from_numpy(np.ascontiguousarray(img0.transpose(2, 0, 1))).float().to(DEVICE) / 255.0
    tensor1 = torch.from_numpy(np.ascontiguousarray(img1.transpose(2, 0, 1))).float().to(DEVICE) / 255.0

    img0 = tensor0.unsqueeze(0)
    img1 = tensor1.unsqueeze(0)

    # Pad to multiple of 32
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    # Run inference — generate single intermediate frame
    middle = model.inference(img0, img1)

    # Convert back to numpy BGR uint8
    result = (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    return result
