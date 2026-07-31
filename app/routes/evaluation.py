import cv2
import numpy as np
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.preprocessing import load_image_from_bytes
from app.evaluation.metrics import evaluate_images
from app.routes.inference import sessions

router = APIRouter()


@router.post("/evaluate")
async def evaluate_frame_quality(
    ground_truth_image: UploadFile = File(..., description="Ground truth middle frame image"),
    generated_image: Optional[UploadFile] = File(None, description="Generated intermediate frame image (optional if session_id is provided)"),
    session_id: Optional[str] = Form(None, description="Session ID of prior pipeline execution (optional if generated_image is uploaded)"),
):
    """
    POST /evaluate
    Compare the generated intermediate frame with the ground truth image.
    Calculates PSNR, SSIM, MSE, and FSIM quality metrics along with quality ratings.
    """
    # 1. Determine Generated Image source
    gen_img_np = None

    if generated_image is not None:
        try:
            bytes_gen = await generated_image.read()
            if len(bytes_gen) > 0:
                gen_img_np = load_image_from_bytes(bytes_gen)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read generated_image: {str(e)}")

    if gen_img_np is None and session_id:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        session = sessions[session_id]
        if not session.get("interpolation_done") or "result_path" not in session:
            raise HTTPException(status_code=400, detail="Intermediate frame has not been generated for this session yet.")
        gen_img_np = cv2.imread(session["result_path"])

    if gen_img_np is None:
        raise HTTPException(
            status_code=400,
            detail="Missing generated intermediate image. Please upload 'generated_image' or provide a valid 'session_id'.",
        )

    # 2. Read Ground Truth Image
    try:
        bytes_gt = await ground_truth_image.read()
        if not bytes_gt:
            raise HTTPException(status_code=400, detail="Ground truth image file is empty.")
        gt_img_np = load_image_from_bytes(bytes_gt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode ground truth image: {str(e)}")

    # 3. Calculate Evaluation Metrics & Ratings
    try:
        results = evaluate_images(gen_img_np, gt_img_np)
        return {
            "status": "success",
            "session_id": session_id,
            "metrics": results["metrics"],
            "ratings": results["ratings"],
            "overall_quality": results["overall_quality"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
