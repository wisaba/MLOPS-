from fastapi import FastAPI, UploadFile, File, HTTPException
from model import predict_image

app = FastAPI(title="Cats vs Dogs Classifier API")


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        result = predict_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing image")

    return {
        "filename": file.filename,
        "prediction": result
    }
