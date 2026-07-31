import cv2
import numpy as np
import os


def visualize_flow(flow, output_path: str) -> str:
    """
    Convert optical flow tensor to an HSV color visualization and save it.
    Original logic from pipeline/flow_visualizer.py.

    Args:
        flow: Flow tensor from RAFT (with batch dimension)
        output_path: Full path to save the flow image

    Returns:
        The output path where the image was saved
    """
    # Remove batch dimension
    flow = flow.squeeze(0)

    # Convert to NumPy
    flow = flow.cpu().numpy()

    # Split into horizontal and vertical motion
    dx = flow[0]
    dy = flow[1]

    # Convert to polar coordinates
    magnitude, angle = cv2.cartToPolar(dx, dy)

    # HSV image
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, flow_image)

    print(f"[OK] Optical flow saved to {output_path}")

    return output_path
