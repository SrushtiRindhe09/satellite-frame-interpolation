# ISRO Satellite Frame Interpolation — Backend

AI-powered backend for generating optical flow visualization and intermediate satellite frames using **RAFT** and **RIFE** deep learning models.

## Project Structure

```
ISRO_Hackathon/
├── app/
│   ├── main.py               # FastAPI application entry point
│   ├── config.py              # Configuration (paths, device, sizes)
│   ├── routes/
│   │   └── inference.py       # API endpoint handlers
│   └── services/
│       ├── preprocessing.py   # Image load, resize, validate, tensor convert
│       ├── raft_inference.py  # RAFT optical flow inference
│       ├── rife_inference.py  # RIFE frame interpolation inference
│       └── flow_visualizer.py # Optical flow → HSV visualization
├── rife/                      # RIFE HDv3 inference-only model files
│   ├── RIFE_HDv3.py
│   ├── IFNet_HDv3.py
│   └── warplayer.py
├── weights/
│   └── rife/
│       └── flownet.pkl        # RIFE pretrained weights
├── outputs/                   # Generated at runtime
│   ├── optical_flow/
│   └── interpolated/
├── uploads/                   # Uploaded images (per session)
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place RIFE weights

Ensure `weights/rife/flownet.pkl` exists. This is the pretrained RIFE HDv3 model.

### 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The server loads both RAFT and RIFE models **once** at startup.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server status |
| `/health` | GET | Health check (model load status) |
| `/upload-images` | POST | Upload Image A and Image B |
| `/generate-optical-flow` | POST | Generate RAFT optical flow |
| `/approve-flow` | POST | Approve the optical flow |
| `/generate-intermediate` | POST | Generate RIFE intermediate frame |
| `/download-result/{type}` | GET | Download result image |
| `/docs` | GET | Interactive Swagger API documentation |

## Workflow

1. **POST** `/upload-images` — Upload two satellite images → get `session_id`
2. **POST** `/generate-optical-flow?session_id=...` — Run RAFT → get flow image
3. *(Frontend shows the flow image to the user)*
4. **POST** `/approve-flow?session_id=...` — User approves the flow
5. **POST** `/generate-intermediate?session_id=...` — Run RIFE → get interpolated frame
6. **GET** `/download-result/interpolated?session_id=...` — Download the result

## Models

- **RAFT** (Recurrent All-Pairs Field Transforms) — Optical flow estimation via `torchvision`
- **RIFE HDv3** (Real-Time Intermediate Flow Estimation) — Frame interpolation

## Attribution

- RAFT: Teed & Deng, ECCV 2020
- RIFE: Huang et al., ECCV 2022
