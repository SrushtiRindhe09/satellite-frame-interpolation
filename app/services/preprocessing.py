import cv2
import numpy as np
from torchvision import transforms

from app.config import TARGET_SIZE


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from raw bytes (uploaded file) into a CV2 numpy array.
    """
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image. Please upload a valid image file.")

    return image


def preprocess_images(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Resize both images to the target size (512x512).
    Original logic from pipeline/image_preprocessor.py.
    """
    image1 = cv2.resize(image1, TARGET_SIZE)
    image2 = cv2.resize(image2, TARGET_SIZE)

    return image1, image2


def validate_images(image1: np.ndarray, image2: np.ndarray) -> None:
    """
    Validate that both images have the same dimensions.
    Original logic from pipeline/validator.py.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image sizes do not match! "
            f"Image 1: {image1.shape}, Image 2: {image2.shape}"
        )


def convert_to_tensor(image1: np.ndarray, image2: np.ndarray) -> tuple:
    """
    Convert preprocessed CV2 images to PyTorch tensors with batch dimension.
    Original logic from pipeline/tensor_converter.py.
    """
    transform = transforms.ToTensor()

    tensor1 = transform(image1)
    tensor2 = transform(image2)

    # Add batch dimension
    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    return tensor1, tensor2
