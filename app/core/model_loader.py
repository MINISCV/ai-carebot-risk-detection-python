from fastapi import Request
from transformers import AutoTokenizer, T5TokenizerFast, T5ForConditionalGeneration

from app.models.service_models import ContextRiskModel
from app.core.config import LABEL_MODEL_PATH, SUMMARY_MODEL_PATH, DEVICE

class ModelManager:
    """
    애플리케이션에서 사용하는 머신러닝 모델과 토크나이저를 로드하고 관리하는 클래스.
    """
    def __init__(self):
        self.device = DEVICE
        
        # 라벨 분류 모델 관련
        self.label_tokenizer = None
        self.label_model = None
        
        # 요약 모델 관련
        self.summary_tokenizer = None
        self.summary_model = None

        self._load_all_models()

    def _load_all_models(self):
        """모든 모델과 토크나이저를 메모리에 로드합니다."""
        # 라벨 분류 모델 로드
        self.label_tokenizer = AutoTokenizer.from_pretrained(LABEL_MODEL_PATH, use_fast=True)
        self.label_model = ContextRiskModel.from_pretrained(LABEL_MODEL_PATH)
        self.label_model.eval()
        self.label_model.to(self.device)

        # 요약 모델 로드
        self.summary_tokenizer = T5TokenizerFast.from_pretrained(SUMMARY_MODEL_PATH)
        self.summary_model = T5ForConditionalGeneration.from_pretrained(SUMMARY_MODEL_PATH)
        self.summary_model.to(self.device)
        
        print("All models loaded successfully.")

def get_model_manager(request: Request) -> ModelManager:
    """FastAPI의 Depends를 통해 ModelManager 인스턴스를 가져옵니다."""
    return request.app.state.model_manager