import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.pipeline import analyze_upload


APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = APP_ROOT / "frontend"
PRIVATE_DATA_ROOT = APP_ROOT / "private-data"

app = FastAPI(
    title="DermaLens Local",
    version="0.1.0",
    description="Local-first facial skin screening prototype with no photo retention by default.",
)

app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "privacy": {
            "save_uploads": os.getenv("SAVE_UPLOADS", "false").lower() == "true",
            "offline_hf": os.getenv("HF_HUB_OFFLINE") == "1",
        },
    }


@app.post("/api/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image.")

    raw = await file.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Use a photo under 12 MB.")

    try:
        return analyze_upload(
            raw,
            original_filename=file.filename or "upload",
            save_uploads=os.getenv("SAVE_UPLOADS", "false").lower() == "true",
            private_data_root=PRIVATE_DATA_ROOT,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8080"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
