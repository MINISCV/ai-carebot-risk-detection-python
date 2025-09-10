from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer

nltk.download('punkt')

# Configs

dl_instances = {}
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load
    model_path = './label_classifier_model'
    dl_instances["classify_tokenizer"] = KoBERTTokenizer.from_pretrained(model_path)
    dl_instances["classify_model"] = BertForSequenceClassification.from_pretrained(model_path)

    model_path = "lcw99/t5-base-korean-text-summary"
    dl_instances["summary_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
    dl_instances["summary_model"] = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    yield
    # Clean up
    # ...

app = FastAPI(lifespan=lifespan)

# origins = [
#     "http://localhost:8080" 
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class Risk(str, Enum):
    POSITIVE = "POSITIVE"
    DANGER = "DANGER"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class Dialogue(BaseModel):
    dollId: str
    text: str
    utteredAt: datetime

label_encoder = joblib.load('./label_classifier_model/label_encoder.joblib')

def label(text: str):
    encoding = dl_instances["classify_tokenizer"].encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = dl_instances["classify_model"](input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)

    predicted_label_encoded = prediction.item()
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]

    return predicted_label

def summarize(text: str):
    prefix = "summarize: "
    inputs = [prefix + text]

    inputs = dl_instances["summary_tokenizer"](inputs, max_length=512, truncation=True, return_tensors="pt")
    output = dl_instances["summary_model"].generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
    decoded_output = dl_instances["summary_tokenizer"].batch_decode(output, skip_special_tokens=True)[0]
    summary = nltk.sent_tokenize(decoded_output.strip())[0]

    return summary


# Routes

@app.get("/")
def index():
    return {
        "message": "hello world"
    }

@app.post("/predict/sentiment")
def predict_sentiment(dialogues: List[Dialogue]):
    for dialogue in dialogues:
        print(dialogue)
    return {
        "message": f"총 {len(dialogues)}개의 데이터가 성공적으로 수신되었습니다."
    }

@app.get("/analyze")
def analyze(text: str = None):
    if not text:
        return {"message": "invalid_request", "result": None}

    return {
        "message": "success",
        "result": {
            "summary": summarize(text),
            "label": label(text),
        }
    }
