import pandas as pd
import re
from typing import Dict
from tqdm import tqdm

# --- 1. 설정 (Configurations) ---

K_CONTEXT = 20 # 라벨 모델 컨텍스트 설정

# --- 2. 라벨 모델 전처리 (Label Model Preprocessing) ---

def parse_datetime_column(df: pd.DataFrame, col: str = "uttered_at") -> pd.DataFrame:
    """DataFrame의 시간 컬럼을 datetime 객체로 변환합니다."""
    df[col] = pd.to_datetime(df[col])
    return df

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 관련 특징(delta_t, hour)을 계산합니다."""
    # API 요청 전체를 단일 문맥으로 간주하므로 session_id에 의존하지 않습니다.
    df['delta_t'] = df.groupby('doll_id')['uttered_at'].diff().dt.total_seconds().fillna(0)
    df["hour"] = df["uttered_at"].dt.hour
    return df

def apply_emotional_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """사전 기반 감정 특징을 추출하여 DataFrame에 추가합니다."""
    def extract_emotional_features(text: str) -> Dict[str, float]:
        features = {}
        text = str(text)
        risk_lexicon = {
            'emergency': {'keywords': ['도와줘', '구해줘', '살려줘', '응급', '위험', '사고', '병원', '119', '112', '불이야', '죽고 싶어', '죽고 싶다', '자살'], 'weight': 3.0},
            'critical': {'keywords': ['아파', '아프다', '고통', '힘들어', '괴롭다', '괴로워', '스트레스', '우울', '불안', '외롭다', '외로워', '쓸쓸하다', '쓸쓸해'], 'weight': 2.0},
            'danger': {'keywords': ['힘들어', '어려워', '괴로워', '스트레스', '우울', '불안', '걱정', '답답하다', '답답해'], 'weight': 1.5},
            'positive': {'keywords': ['좋아', '행복', '기뻐', '만족', '감사', '고마워'], 'weight': 0.5}
        }
        for category, data in risk_lexicon.items():
            score = 0
            count = 0
            for keyword in data['keywords']:
                if keyword in text:
                    c = text.count(keyword)
                    score += c * data['weight']
                    count += c
            features[f'emo_{category}_score'] = score
            features[f'emo_{category}_count'] = count
        return features

    emo_feats = [extract_emotional_features(t) for t in df[text_col].fillna("")]
    emo_df = pd.DataFrame(emo_feats, index=df.index)
    for col in emo_df.columns:
        if col.startswith('emo_'):
            emo_df[col] = pd.to_numeric(emo_df[col], errors='coerce').fillna(0).astype(float)
    return pd.concat([df, emo_df], axis=1)

def build_context_sequences(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """K개의 이전 대화를 포함하는 컨텍스트 시퀀스를 생성합니다. API 요청 전체를 단일 문맥으로 처리합니다."""
    df = df.sort_values(["doll_id", "uttered_at"]).reset_index(drop=True)
    seq_data = []
    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    for _, group in df.groupby(["doll_id"], sort=False):
        texts = group['text'].tolist()
        delta_ts = group['delta_t'].tolist()
        hours = group['hour'].tolist()
        emo_vectors = group[emo_cols].values.tolist()
        for i in range(len(group)):
            start = max(0, i - k + 1)
            seq_data.append({
                'seq_texts': texts[start:i+1],
                'seq_delta_t': delta_ts[start:i+1],
                'seq_hours': hours[start:i+1],
                'seq_emo_vectors': emo_vectors[start:i+1],
            })
    seq_df = pd.DataFrame(seq_data)
    return pd.concat([df.reset_index(drop=True), seq_df], axis=1)


# --- 3. 요약 모델 전처리 (Summary Model Preprocessing) ---

def clean_text_for_summary(text: str) -> str:
    """요약 모델을 위한 대화 문자열 전처리"""
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()