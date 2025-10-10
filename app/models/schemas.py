from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

# --- Pydantic Schemas ---
class Dialogue(BaseModel):
    """API 요청의 개별 대화 기록"""
    doll_id: str
    text: str = Field(..., min_length=1)
    uttered_at: datetime

    class Config:
        populate_by_name = True

class ConfidenceScores(BaseModel):
    """위험도 분류 모델의 신뢰도 점수"""
    positive: str
    danger: str
    critical: str
    emergency: str

class DialogueResult(BaseModel):
    """개별 대화 분석 결과"""
    seq: int
    doll_id: str
    text: str
    uttered_at: datetime
    label: str
    confidence_scores: ConfidenceScores

class Evidence(BaseModel):
    """종합 분석 결과의 근거가 되는 대화"""
    seq: int
    text: str
    score: str

class Reason(BaseModel):
    """종합 분석 결과의 판단 근거 (증거+요약)"""
    evidence: List[Evidence]
    summary: str

class OverallResult(BaseModel):
    """전체 대화에 대한 종합 분석 결과"""
    doll_id: str
    dialogue_count: int
    char_length: int
    label: str
    confidence_scores: ConfidenceScores
    full_text: str
    reason: Reason

class AnalysisResponse(BaseModel):
    """/analyze API의 최종 응답 모델"""
    overall_result: OverallResult
    dialogue_result: List[DialogueResult]