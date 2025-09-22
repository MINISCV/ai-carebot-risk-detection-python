import re
from datetime import datetime
from typing import List

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
# from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


# ---------------------------
# Configs & Declaration
# ---------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class SentimentClassifier(nn.Module):
    """라벨 분류 모델에 필요한 분류기"""

    def __init__(self, model_name, n_classes, dropout_rate):
        super(SentimentClassifier, self).__init__()
        self.bart = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bart.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.fc(output)
        return logits

def load_label_model(models):
    """라벨 분류 모델 로드 - klue/roberta-base 전이학습"""

    state = torch.load("./label_model/finetuned_klue_roberta_base.pth", map_location=device)
    models["label_tokenizer"] = AutoTokenizer.from_pretrained("./label_model/klue_roberta_base_tokenizer")
    models["label_model"] = SentimentClassifier(model_name="klue/roberta-base", n_classes=4, dropout_rate=0.2)
    models["label_model"].load_state_dict(state)
    models["label_model"].eval()
    models["label_model"].to(device)

def load_summary_model(models):
    """요약 모델 로드 - skt/kobart-base-v1 전이학습"""

    model_path = "./summary_model"
    models["summary_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
    models["summary_model"] = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    models["summary_model"].eval()
    models["summary_model"].to(device)

# 로드한 모델 인스턴스 저장 딕셔너리
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리 - 웹 서비스 실행 시 모델 로드"""

    # Load
    load_label_model(models)
    load_summary_model(models)
    yield
    # Clean up
    # ...

# FastAPI 객체
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

def clean_text(text):
    """대화 문자열 전처리"""

    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def label(text, model, tokenizer, max_length=128):
    """라벨 분류 (positive, danger, critical, emergency)"""

    cleaned_text = clean_text(text)
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    confience_scores = [format(round(n, 4), '.4f') for n in probabilities[0].tolist()]
    return label_domain[predicted_class], dict(zip(label_domain, confience_scores))

def summarize(text, model, tokenizer, max_length=1024):
    """대화 요약"""

    text = clean_text(text)
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_text_length = len(tokenizer.encode(text))

    # TODO 요약 간소화
    current_min_length = 5
    if input_text_length <= 5:
        current_max_length = 10
    elif input_text_length <= 20:
        current_max_length = 20
    elif input_text_length <= 64:
        current_max_length = 50
    else:
        current_max_length = input_text_length
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=current_max_length,
            min_length=current_min_length,
            num_beams=4,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary

def validate(dialogues):
    """유효성 검사"""

    if len(dialogues) == 0:
        return "failure", "empty_list"
    
    if len({dialogue.dollId for dialogue in dialogues}) != 1:
        return "failure", "invalid_doll_id"
    
    return "success", ""

class Dialogue(BaseModel):
    """개별 대화 기록"""

    dollId: str
    text: str
    utteredAt: datetime

# 라벨 분류 매핑 목록
label_domain = ['positive', 'danger', 'critical', 'emergency']


# ---------------------------
# Routes
# ---------------------------

@app.post("/analyze")
def analyze(dialogues: List[Dialogue]):
    """대화 분석 요청 API"""

    validation = validate(dialogues)
    if validation[0] == "failure":
        content = {
            "result": "failure",
            "validation_msg": validation[1],
        }
        return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    
    dialogue_result = []
    full_text = ""

    for seq, dialogue in enumerate(dialogues):
        labeled = label(dialogue.text, models["label_model"], models["label_tokenizer"])
        dialogue_result.append({
            "seq": seq,
            "doll_id": dialogue.dollId,
            "text": dialogue.text,
            "uttered_at": dialogue.utteredAt,
            "label": labeled[0],
            "confidence_scores": labeled[1],
        })
        full_text += " " + dialogue.text
    
    full_text = full_text.strip()
    overall_label = label(full_text, models["label_model"], models["label_tokenizer"])
    summary = summarize(full_text, models["summary_model"], models["summary_tokenizer"])
    evidences = sorted(dialogue_result, key=lambda x: float(x["confidence_scores"][overall_label[0]]), reverse=True)
    evidences = [{"seq": v["seq"], "text": v["text"], "score": v["confidence_scores"][overall_label[0]]} for v in evidences][:2]

    return {
        "result": "success",
        "validation_msg": "",
        "overall_result": {
            "doll_id" : dialogues[0].dollId,
            "dialogue_count" : len(dialogues),
            "char_length" : len(full_text),
            "label" : overall_label[0],
            "confidence_scores" : overall_label[1],
            "full_text" : full_text,
            "reason": {
                "evidence": evidences,
                "summary" : summary,
            },
        },
        "dialogue_result": dialogue_result,
    }
