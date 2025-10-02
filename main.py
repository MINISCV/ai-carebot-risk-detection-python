import re
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing_extensions import Annotated
# from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5TokenizerFast, T5ForConditionalGeneration

from service_models import ContextRiskModel, ContextDataset, collate_fn
from preprocessing import (
    parse_datetime_column, compute_time_features, apply_emotional_features,
    build_context_sequences, clean_text_for_summary, K_CONTEXT
)

# ---------------------------
# Configs & Declaration
# ---------------------------

# --- Configuration ---
LABEL_MODEL_PATH = "./label_model"
SUMMARY_MODEL_PATH = "./summary_model"
LABEL_ORDER = ['positive', 'danger', 'critical', 'emergency']
EVIDENCE_COUNT = 2

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def load_label_model(models):
    """라벨 분류 모델 로드"""
    models["label_tokenizer"] = AutoTokenizer.from_pretrained(LABEL_MODEL_PATH, use_fast=True)
    models["label_model"] = ContextRiskModel.from_pretrained(LABEL_MODEL_PATH)
    models["label_model"].eval()
    models["label_model"].to(device)

def load_summary_model(models):
    """요약 모델 로드"""
    models["summary_tokenizer"] = T5TokenizerFast.from_pretrained(SUMMARY_MODEL_PATH)
    models["summary_model"] = T5ForConditionalGeneration.from_pretrained(SUMMARY_MODEL_PATH)
    models["summary_model"].to(device)

# 로드한 모델 인스턴스 저장 딕셔너리
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리 - 웹 서비스 실행 시 모델 로드"""
    load_label_model(models)
    load_summary_model(models)
    yield

# FastAPI 객체
app = FastAPI(lifespan=lifespan)

def summarize(text, model, tokenizer, max_length=256):
    """대화 요약"""
    cleaned_text = clean_text_for_summary(text)
    input_ids = tokenizer.encode("summarize: " + cleaned_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- Pydantic Schemas ---
class Dialogue(BaseModel):
    """개별 대화 기록"""
    doll_id: str
    text: str = Field(..., min_length=1)
    uttered_at: datetime

    class Config:
        populate_by_name = True

class ConfidenceScores(BaseModel):
    positive: str
    danger: str
    critical: str
    emergency: str

class DialogueResult(BaseModel):
    seq: int
    doll_id: str
    text: str
    uttered_at: datetime
    label: str
    confidence_scores: ConfidenceScores

class Evidence(BaseModel):
    seq: int
    text: str
    score: str

class Reason(BaseModel):
    evidence: List[Evidence]
    summary: str

class OverallResult(BaseModel):
    doll_id: str
    dialogue_count: int
    char_length: int
    label: str
    confidence_scores: ConfidenceScores
    full_text: str
    reason: Reason

class AnalysisResponse(BaseModel):
    overall_result: OverallResult
    dialogue_result: List[DialogueResult]

# ---------------------------
# Routes
# ---------------------------

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(dialogues: Annotated[List[Dialogue], Body()]):
    """대화 분석 요청 API"""
    # --- Validation ---
    if not dialogues:
        return JSONResponse(content={"validation_msg": "empty_list"}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    if len({d.doll_id for d in dialogues}) != 1:
        return JSONResponse(content={"validation_msg": "invalid_doll_id"}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    # --- Label Model Preprocessing ---
    df = pd.DataFrame([d.model_dump() for d in dialogues])
    df = parse_datetime_column(df)
    df = compute_time_features(df)
    df = apply_emotional_features(df)
    df = build_context_sequences(df, k=K_CONTEXT)

    # --- Tokenization for Label Model ---
    label_tokenizer = models["label_tokenizer"]
    sep = label_tokenizer.sep_token
    df['joined_text'] = [f" {sep} ".join(str(t) for t in texts) for texts in df['seq_texts']]
    
    # 토크나이징 수행
    encodings = label_tokenizer(
        df['joined_text'].tolist(),
        truncation=True, 
        padding=False,
        max_length=label_tokenizer.model_max_length
    )
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']

    # --- Label Model Prediction ---    
    label_model = models["label_model"]
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}

    # 전처리된 DataFrame으로 데이터셋 생성
    dataset = ContextDataset(df, label_map)
    pad_token_id = label_tokenizer.pad_token_id if label_tokenizer.pad_token_id is not None else 0
    dataloader = DataLoader(dataset, batch_size=len(df), shuffle=False, collate_fn=lambda b: collate_fn(b, pad_token_id))
    
    batch = next(iter(dataloader))
    inputs = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = label_model(**inputs)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_ids = torch.argmax(probabilities, dim=-1)

    # --- Format Results ---
    dialogue_result = []
    full_text = ""
    for i in range(len(df)):
        pred_class_id = predicted_class_ids[i].item()
        label_name = LABEL_ORDER[pred_class_id]
        confidence = {label: f"{score:.4f}" for label, score in zip(LABEL_ORDER, probabilities[i].tolist())}
        
        dialogue_result.append({
            "seq": i,
            "doll_id": dialogues[i].doll_id,
            "text": dialogues[i].text,
            "uttered_at": dialogues[i].uttered_at,
            "label": label_name,
            "confidence_scores": confidence,
        })
        full_text += " " + dialogues[i].text
    
    full_text = full_text.strip()

    # --- Overall Analysis ---
    risk_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    overall_result_item = max(dialogue_result, key=lambda x: risk_map[x['label']])
    overall_label_name = overall_result_item['label']
    overall_confidence = overall_result_item['confidence_scores']

    summary = summarize(full_text, models["summary_model"], models["summary_tokenizer"])
    
    evidences = sorted(dialogue_result, key=lambda x: float(x["confidence_scores"][overall_label_name]), reverse=True)
    evidences = [{"seq": v["seq"], "text": v["text"], "score": v["confidence_scores"][overall_label_name]} for v in evidences][:EVIDENCE_COUNT]

    return {
        "overall_result": {
            "doll_id" : dialogues[0].doll_id,
            "dialogue_count" : len(dialogues),
            "char_length" : len(full_text),
            "label" : overall_label_name,
            "confidence_scores" : overall_confidence,
            "full_text" : full_text,
            "reason": {
                "evidence": evidences,
                "summary" : summary,
            },
        },
        "dialogue_result": dialogue_result,
    }
