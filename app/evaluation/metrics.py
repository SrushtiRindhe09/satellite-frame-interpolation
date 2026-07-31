import cv2
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity as skimage_ssim,
    mean_squared_error as skimage_mse,
)


def compute_fsim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Feature Similarity Index (FSIM) using gradient magnitude
    and luminance similarity maps.
    """
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2

    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)

    # Compute Sobel Gradients
    gx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    g1 = np.sqrt(gx1**2 + gy1**2)

    gx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    g2 = np.sqrt(gx2**2 + gy2**2)

    # Gradient Similarity map
    T1 = 160.0
    S_g = (2 * g1 * g2 + T1) / (g1**2 + g2**2 + T1)

    # Luminance Similarity map
    T2 = 0.85
    S_l = (2 * gray1 * gray2 + T2) / (gray1**2 + gray2**2 + T2)

    # Weight by maximum gradient magnitude
    weight = np.maximum(g1, g2) + 1e-4
    fsim_val = np.sum(S_g * S_l * weight) / np.sum(weight)

    return float(np.clip(fsim_val, 0.0, 1.0))


def rate_psnr(val: float) -> str:
    if val >= 40.0:
        return "Excellent"
    elif val >= 35.0:
        return "Very Good"
    elif val >= 30.0:
        return "Good"
    else:
        return "Poor"


def rate_ssim(val: float) -> str:
    if val >= 0.99:
        return "Excellent"
    elif val >= 0.95:
        return "Very Good"
    elif val >= 0.90:
        return "Good"
    else:
        return "Poor"


def rate_mse(val: float) -> str:
    if val <= 5.0:
        return "Excellent"
    elif val <= 20.0:
        return "Very Good"
    elif val <= 100.0:
        return "Good"
    else:
        return "Poor"


def rate_fsim(val: float) -> str:
    if val >= 0.99:
        return "Excellent"
    elif val >= 0.95:
        return "Very Good"
    elif val >= 0.90:
        return "Good"
    else:
        return "Poor"


def compute_overall_quality(ratings: dict) -> str:
    score_map = {
        "Excellent": 4.0,
        "Very Good": 3.0,
        "Good": 2.0,
        "Poor": 1.0,
    }
    scores = [score_map[r] for r in ratings.values() if r in score_map]
    if not scores:
        return "Unknown"
    avg = sum(scores) / len(scores)

    if avg >= 3.5:
        return "Excellent"
    elif avg >= 2.8:
        return "Very Good"
    elif avg >= 2.0:
        return "Good"
    else:
        return "Poor"


def evaluate_images(generated_img: np.ndarray, ground_truth_img: np.ndarray) -> dict:
    """
    Compare generated intermediate image against ground truth image and calculate
    PSNR, SSIM, MSE, and FSIM along with quality interpretation ratings.
    """
    if generated_img is None or ground_truth_img is None:
        raise ValueError("One or both images could not be loaded for evaluation.")

    # Automatically resize ground truth image to match generated image dimensions if needed
    h, w = generated_img.shape[:2]
    if ground_truth_img.shape[:2] != (h, w):
        ground_truth_img = cv2.resize(ground_truth_img, (w, h))

    # Compute PSNR
    psnr_val = float(skimage_psnr(ground_truth_img, generated_img, data_range=255))
    psnr_val = round(psnr_val, 2)

    # Compute SSIM
    channel_axis = 2 if generated_img.ndim == 3 else None
    ssim_val = float(skimage_ssim(ground_truth_img, generated_img, channel_axis=channel_axis, data_range=255))
    ssim_val = round(ssim_val, 4)

    # Compute MSE
    mse_val = float(skimage_mse(ground_truth_img, generated_img))
    mse_val = round(mse_val, 2)

    # Compute FSIM
    fsim_val = compute_fsim(generated_img, ground_truth_img)
    fsim_val = round(fsim_val, 4)

    # Individual Ratings
    ratings = {
        "psnr": rate_psnr(psnr_val),
        "ssim": rate_ssim(ssim_val),
        "mse": rate_mse(mse_val),
        "fsim": rate_fsim(fsim_val),
    }

    # Overall Quality Rating
    overall = compute_overall_quality(ratings)

    return {
        "metrics": {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "mse": mse_val,
            "fsim": fsim_val,
        },
        "ratings": ratings,
        "overall_quality": overall,
    }
