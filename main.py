from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import List

app = FastAPI()

class Risk(str, Enum):
    POSITIVE = "POSITIVE"
    DANGER = "DANGER"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class Dialogue(BaseModel):
    dollId: str
    text: str
    utteredAt: datetime
    
@app.post("/predict/sentiment")
def predict_sentiment(dialogues: List[Dialogue]):
    for dialogue in dialogues:
        print(dialogue)
    return {
        "message": f"총 {len(dialogues)}개의 데이터가 성공적으로 수신되었습니다."
    }