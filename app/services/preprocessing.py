import pandas as pd
import re
from typing import Dict

# --- 라벨 모델 전처리 (Label Model Preprocessing) ---

def parse_datetime_column(df: pd.DataFrame, col: str = "uttered_at") -> pd.DataFrame:
    """DataFrame의 시간 컬럼을 datetime 객체로 변환합니다."""
    df[col] = pd.to_datetime(df[col])
    return df

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간 관련 특성(delta_t, hour)을 계산하여 DataFrame에 추가합니다.
    - delta_t: 이전 발화와의 시간 차이(초). 세션 시작 발화는 0으로 설정됩니다.
    - hour: 발화가 발생한 시간(0-23).
    """
    # API 요청 전체를 단일 문맥으로 간주하므로 session_id에 의존하지 않습니다.
    df['delta_t'] = df.groupby('doll_id')['uttered_at'].diff().dt.total_seconds().fillna(0)
    df["hour"] = df["uttered_at"].dt.hour
    return df

def apply_emotional_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    텍스트 기반의 감정/위험 어휘 사전을 사용하여 감정적 특성을 추출하고 점수화합니다.
    미리 정의된 키워드의 등장 횟수와 가중치를 기반으로 점수를 계산하여 모델의 특성으로 활용합니다.

    Args:
        df (pd.DataFrame): 원본 DataFrame.
        text_col (str): 감정 특성을 추출할 텍스트 컬럼 이름.

    Returns:
        pd.DataFrame: 감정 특성 점수 컬럼(emo_*)이 추가된 DataFrame.
    """
    def extract_emotional_features(text: str) -> Dict[str, float]:
        """단일 텍스트에서 감정 관련 키워드를 찾아 점수를 계산합니다."""
        features = {}
        text = str(text)
        # 위험도별 키워드와 가중치 사전
        risk_lexicon = {
            'emergency': {'keywords': ['도와줘', '구해줘', '살려줘', '응급', '위험', '사고', '병원', '119', '112', '불이야', '죽고 싶어', '죽고 싶다', '자살'], 'weight': 3.0},
            'critical': {'keywords': ['아파', '아프다', '고통', '괴롭다', '괴로워', '우울', '외롭다', '외로워', '쓸쓸'], 'weight': 2.0},
            'danger': {'keywords': ['힘들어', '어려워', '스트레스', '불안', '걱정', '답답'], 'weight': 1.5},
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

    # DataFrame의 모든 텍스트에 대해 감정 특성 추출을 적용합니다.
    emo_feats = [extract_emotional_features(t) for t in df[text_col].fillna("")]
    emo_df = pd.DataFrame(emo_feats, index=df.index)
    
    # 생성된 특성 컬럼들의 데이터 타입을 숫자로 변환합니다.
    for col in emo_df.columns:
        if col.startswith('emo_'):
            emo_df[col] = pd.to_numeric(emo_df[col], errors='coerce').fillna(0).astype(float)
            
    # 원본 DataFrame에 감정 특성 DataFrame을 합칩니다.
    return pd.concat([df, emo_df], axis=1)

def build_context_sequences(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    각 발화를 기준으로, 이전 k개의 발화를 포함하는 문맥 시퀀스를 생성합니다.
    모델이 현재 발화뿐만 아니라 이전 대화의 흐름을 이해하도록 돕습니다.
    API 요청 전체를 단일 문맥(세션)으로 간주하고 처리합니다.
    """
    # 사용자 ID와 발화 시간으로 정렬하여 순서를 보장합니다.
    df = df.sort_values(["doll_id", "uttered_at"]).reset_index(drop=True)
    seq_data = []
    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    for _, group in df.groupby(["doll_id"], sort=False):
        texts = group['text'].tolist()
        delta_ts = group['delta_t'].tolist()
        hours = group['hour'].tolist()
        emo_vectors = group[emo_cols].values.tolist()
        
        # 그룹 내 각 발화에 대해 문맥 시퀀스를 생성합니다.
        for i in range(len(group)):
            start_index = max(0, i - k + 1)
            seq_data.append({
                'seq_texts': texts[start_index:i+1],
                'seq_delta_t': delta_ts[start_index:i+1],
                'seq_hours': hours[start_index:i+1],
                'seq_emo_vectors': emo_vectors[start_index:i+1],
            })
    seq_df = pd.DataFrame(seq_data)
    return pd.concat([df.reset_index(drop=True), seq_df], axis=1)


# --- 요약 모델 전처리 (Summary Model Preprocessing) ---

def clean_text_for_summary(text: str) -> str:
    """
    요약 모델 학습에 불필요한 특수문자를 제거하고 공백을 정규화하여 텍스트를 정리합니다.
    한글, 영어, 숫자, 그리고 기본적인 공백 문자만 남깁니다.

    Args:
        text (str): 전처리할 원본 문자열.

    Returns:
        str: 특수문자와 불필요한 공백이 제거된 문자열.
    """
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()