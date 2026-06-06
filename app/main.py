from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from app.model import get_classifier
from app.schemas import Prediction, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()
    yield


app = FastAPI(title="AvesLens Model API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="file must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    predictions = get_classifier().predict(image)

    return PredictResponse(predictions=[Prediction(**p) for p in predictions])
