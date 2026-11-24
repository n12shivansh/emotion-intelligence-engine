from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(
    title="Emotion Intelligence Engine",
    description="Emotion classification using fine-tuned DistilBERT",
    version="1.0"
)

MODEL_PATH = "n12shivansh/emotion-intelligence-engine"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN
)

model.eval()

device = torch.device("cpu")
model.to(device)


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Emotion Intelligence Engine API is running!"
    }


@app.post("/predict")
def predict_emotion(data: TextInput):

    if not data.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_class = torch.argmax(
        probabilities, dim=1
    ).item()

    predicted_emotion = model.config.id2label[predicted_class]

    confidence = probabilities[0][predicted_class].item()

    return {
        "input_text": data.text,
        "predicted_emotion": predicted_emotion,
        "confidence": round(confidence, 4)
    }