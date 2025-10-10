from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from typing import List
from app.models.schemas import Dialogue, AnalysisResponse
from app.services.analysis import analyze_dialogues
from app.core.model_loader import ModelManager, get_model_manager

# ---------------------------
# Init
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리 - 웹 서비스 실행 시 모델 로드"""
    # ModelManager 인스턴스를 생성하여 모델을 로드합니다.
    app.state.model_manager = ModelManager()
    yield
    # 정리(cleanup) 로직이 필요하다면 여기에 추가합니다.
    app.state.model_manager = None

app = FastAPI(lifespan=lifespan)

# ---------------------------
# Routes
# ---------------------------

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(dialogues: List[Dialogue], model_manager: ModelManager = Depends(get_model_manager)):
    """대화 분석 요청 API"""
    return analyze_dialogues(dialogues, model_manager)
