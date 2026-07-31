import os
import uuid
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import UPLOADS_DIR, OPTICAL_FLOW_DIR, INTERPOLATED_DIR
from app.services.preprocessing import (
    load_image_from_bytes,
    preprocess_images,
    validate_images,
    convert_to_tensor,
)
from app.services.raft_inference import load_raft_model, predict_optical_flow
from app.services.rife_inference import load_rife_model, generate_intermediate_frame
from app.services.flow_visualizer import visualize_flow

router = APIRouter()

# ---------------------------------------------------------------
# In-memory session store (adequate for hackathon demo)
# ---------------------------------------------------------------
sessions = {}


def get_raft_model(request: Request):
    if not hasattr(request.app.state, "raft_model") or request.app.state.raft_model is None:
        print("[INFO] Lazy loading RAFT model...")
        request.app.state.raft_model = load_raft_model()
    return request.app.state.raft_model


def get_rife_model(request: Request):
    if not hasattr(request.app.state, "rife_model") or request.app.state.rife_model is None:
        print("[INFO] Lazy loading RIFE model...")
        request.app.state.rife_model = load_rife_model()
    return request.app.state.rife_model




@router.post("/process")
async def process_pipeline(
    request: Request,
    image_a: UploadFile = File(..., description="First satellite image"),
    image_b: UploadFile = File(..., description="Second satellite image"),
):
    """
    Unified end-to-end endpoint:
    Upload Image A and Image B -> Backend runs Preprocessing, RAFT Optical Flow, 
    and RIFE Intermediate Frame Generation in sequence -> Returns URLs for both outputs.
    """
    session_id = str(uuid.uuid4())

    # Read uploaded file bytes
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    # Decode images
    img_a = load_image_from_bytes(bytes_a)
    img_b = load_image_from_bytes(bytes_b)

    # Save raw uploads
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    cv2.imwrite(os.path.join(session_dir, "image_a.png"), img_a)
    cv2.imwrite(os.path.join(session_dir, "image_b.png"), img_b)

    try:
        # 1. Preprocess & validate
        img_a_prep, img_b_prep = preprocess_images(img_a.copy(), img_b.copy())
        validate_images(img_a_prep, img_b_prep)
        tensor_a, tensor_b = convert_to_tensor(img_a_prep, img_b_prep)

        # 2. RAFT Optical Flow Inference
        print(f"\n[Session {session_id[:8]}] Running RAFT optical flow...")
        raft_model = get_raft_model(request)
        flow_predictions = predict_optical_flow(raft_model, tensor_a, tensor_b)
        
        flow_path = os.path.join(OPTICAL_FLOW_DIR, f"{session_id}_flow.png")
        visualize_flow(flow_predictions[-1], flow_path)

        # 3. RIFE Intermediate Frame Inference
        print(f"[Session {session_id[:8]}] Running RIFE frame interpolation...")
        rife_model = get_rife_model(request)
        result_frame = generate_intermediate_frame(rife_model, img_a, img_b)
        
        result_path = os.path.join(INTERPOLATED_DIR, f"{session_id}_frame.png")
        cv2.imwrite(result_path, result_frame)

        # Store session data
        sessions[session_id] = {
            "image_a": img_a,
            "image_b": img_b,
            "flow_generated": True,
            "flow_path": flow_path,
            "interpolation_done": True,
            "result_path": result_path,
        }

        print(f"[Session {session_id[:8]}] [OK] Full pipeline execution completed")

        return {
            "session_id": session_id,
            "message": "AI pipeline executed successfully",
            "optical_flow_image": f"/download-result/optical_flow?session_id={session_id}",
            "interpolated_image": f"/download-result/interpolated?session_id={session_id}",
        }
    except Exception as e:
        import traceback
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"[Session {session_id[:8]}] [ERROR] {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/upload-images")

async def upload_images(
    request: Request,
    image_a: UploadFile = File(..., description="First satellite image"),
    image_b: UploadFile = File(..., description="Second satellite image"),
):
    """
    Step 1 & 2: Upload two satellite images.
    Returns a session_id for subsequent API calls.
    """
    session_id = str(uuid.uuid4())

    # Read uploaded file bytes
    bytes_a = await image_a.read()
    bytes_b = await image_b.read()

    # Decode images
    img_a = load_image_from_bytes(bytes_a)
    img_b = load_image_from_bytes(bytes_b)

    # Save raw uploads for reference
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    path_a = os.path.join(session_dir, "image_a.png")
    path_b = os.path.join(session_dir, "image_b.png")
    cv2.imwrite(path_a, img_a)
    cv2.imwrite(path_b, img_b)

    # Store session data
    sessions[session_id] = {
        "image_a": img_a,
        "image_b": img_b,
        "flow_approved": False,
        "flow_generated": False,
        "interpolation_done": False,
    }

    return {
        "session_id": session_id,
        "message": "Images uploaded successfully",
        "image_a_shape": list(img_a.shape),
        "image_b_shape": list(img_b.shape),
    }


@router.post("/generate-optical-flow")
async def generate_optical_flow(request: Request, session_id: str):
    """
    Step 4: Run RAFT to generate optical flow visualization.
    Returns the optical flow image path.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload images first.")

    session = sessions[session_id]
    raft_model = get_raft_model(request)

    # Step 3: Preprocess
    img_a, img_b = preprocess_images(session["image_a"].copy(), session["image_b"].copy())
    validate_images(img_a, img_b)
    tensor_a, tensor_b = convert_to_tensor(img_a, img_b)

    print(f"\n[Session {session_id[:8]}] Running RAFT optical flow...")

    # Step 4: RAFT inference
    flow_predictions = predict_optical_flow(raft_model, tensor_a, tensor_b)

    # Save flow visualization
    flow_path = os.path.join(OPTICAL_FLOW_DIR, f"{session_id}_flow.png")
    visualize_flow(flow_predictions[-1], flow_path)

    # Store flow data in session for RIFE step
    session["flow_predictions"] = flow_predictions
    session["flow_generated"] = True
    session["flow_path"] = flow_path

    print(f"[Session {session_id[:8]}] [OK] Optical flow generated")

    return {
        "session_id": session_id,
        "message": "Optical flow generated successfully",
        "flow_image": f"/download-result/optical_flow?session_id={session_id}",
    }


@router.post("/approve-flow")
async def approve_flow(session_id: str):
    """
    Step 6: Frontend sends approval after user reviews the optical flow.
    Only after this can the intermediate frame be generated.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]

    if not session["flow_generated"]:
        raise HTTPException(status_code=400, detail="Optical flow has not been generated yet.")

    session["flow_approved"] = True

    return {
        "session_id": session_id,
        "message": "Optical flow approved. You can now generate the intermediate frame.",
    }


@router.post("/generate-intermediate")
async def generate_intermediate(request: Request, session_id: str):
    """
    Step 7 & 8: Generate the intermediate frame using RIFE.
    Requires prior flow approval.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]

    if not session["flow_approved"]:
        raise HTTPException(
            status_code=400,
            detail="Optical flow must be approved before generating intermediate frame.",
        )

    rife_model = get_rife_model(request)

    print(f"\n[Session {session_id[:8]}] Running RIFE frame interpolation...")

    # Step 7: RIFE inference with original images
    result_frame = generate_intermediate_frame(
        rife_model,
        session["image_a"],
        session["image_b"],
    )

    # Save result
    result_path = os.path.join(INTERPOLATED_DIR, f"{session_id}_frame.png")
    cv2.imwrite(result_path, result_frame)

    session["interpolation_done"] = True
    session["result_path"] = result_path

    print(f"[Session {session_id[:8]}] [OK] Intermediate frame generated")

    return {
        "session_id": session_id,
        "message": "Intermediate frame generated successfully",
        "interpolated_image": f"/download-result/interpolated?session_id={session_id}",
    }


@router.get("/download-result/{result_type}")
async def download_result(result_type: str, session_id: str):
    """
    Download the generated optical flow or interpolated frame image.
    result_type: 'optical_flow' or 'interpolated'
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]

    if result_type == "optical_flow":
        if not session.get("flow_generated"):
            raise HTTPException(status_code=400, detail="Optical flow not generated yet.")
        file_path = session["flow_path"]

    elif result_type == "interpolated":
        if not session.get("interpolation_done"):
            raise HTTPException(status_code=400, detail="Intermediate frame not generated yet.")
        file_path = session["result_path"]

    else:
        raise HTTPException(status_code=400, detail="Invalid result type. Use 'optical_flow' or 'interpolated'.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found on disk.")

    return FileResponse(
        file_path,
        media_type="image/png",
        filename=os.path.basename(file_path),
    )
