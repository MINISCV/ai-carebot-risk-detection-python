import os
import json
import math
import pandas as pd # type: ignore
import numpy as np # type: ignore
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from transformers import AutoModel
from typing import List, Dict, Any

# --- 1. 데이터셋 클래스 및 함수 (Dataset Classes and Functions) ---

class ContextDataset(Dataset):
    """
    전처리된 데이터를 모델 학습에 사용할 수 있는 형태로 변환하는 PyTorch Dataset 클래스.
    텍스트 데이터 외에 시간, 감정, 문맥 기반의 추가 특성을 생성합니다.
    """
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int]):
        """
        Args:
            df (pd.DataFrame): 전처리 및 파싱이 완료된 DataFrame.
            label_map (Dict[str, int]): 레이블 문자열을 정수 인덱스로 매핑하는 딕셔너리.
        """
        self.df = df
        self.label_map = label_map
        # 감정 특성 관련 컬럼 이름을 미리 추출하여 사용합니다.
        self.emo_cols = [c for c in df.columns if c.startswith("emo_")]
        # 레이블별 위험도 점수를 정의합니다.
        self.risk_scores_by_label = {label: i for i, label in enumerate(LABEL_ORDER)}
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        하나의 데이터 샘플(발화)에 대한 모델 입력값을 생성합니다.
        
        Args:
            idx (int): 가져올 데이터의 인덱스.

        Returns:
            Dict[str, Any]: 모델 입력으로 사용될 텐서 딕셔너리.
        """
        row = self.df.iloc[idx]
        
        # 1. 토크나이징된 텍스트 데이터
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]

        # 2. 시간 관련 특성
        delta_t_val = row["delta_t"]
        last_delta = math.log1p(delta_t_val)
        is_session_start = 1.0 if delta_t_val == 0 else 0.0
        last_hour = row["hour"]
        # 시간(hour)을 순환적인 특성으로 변환하여 23시와 0시가 가깝다는 것을 표현
        hour_sin = math.sin(2 * math.pi * last_hour / 24)
        hour_cos = math.cos(2 * math.pi * last_hour / 24)
        
        # 3. 감정 어휘 기반 특성
        emo_vec = row[self.emo_cols].values.astype(np.float32)

        # 4. 문맥 기반 위험도 특성 (Contextual Risk Feature)
        # 이전 대화들의 위험도와 시간 경과를 함께 고려한 특성입니다.
        # 최근에 위험한 발화가 많았을수록 높은 값을 가집니다.
        seq_labels = row["seq_labels"]
        seq_delta_t = row["seq_delta_t"]
        
        weighted_context_risk = 0.0
        # 문맥에 2개 이상의 발화가 있을 때만 계산 (현재 발화 제외)
        if len(seq_labels) > 1:
            # 현재 발화를 제외한 이전 발화들에 대해 반복
            for i in range(len(seq_labels) - 1):
                label = seq_labels[i]
                delta_t = seq_delta_t[i+1]  # 해당 발화와 다음 발화 사이의 시간 간격
                
                # 이전 발화의 실제 레이블을 기반으로 위험 점수를 가져옵니다.
                utterance_risk_score = self.risk_scores_by_label.get(label, 0)
                
                # 위험 점수가 0보다 클 경우, 시간 경과(delta_t)에 따라 지수적으로 점수를 감쇠시킴
                if utterance_risk_score > 0:
                    # lambda는 감쇠율을 조절하는 하이퍼파라미터. 최근 발화일수록 더 큰 영향을 줌.
                    # 10분(600초)이 지나면 영향력이 약 10%로 감소(90% 감소)하는 수준입니다. exp(-0.00384 * 600) ~= 0.1
                    decay_lambda = 0.00384
                    weighted_context_risk += utterance_risk_score * math.exp(-decay_lambda * delta_t)
        
        # 최종 문맥 위험도 점수에 log1p를 적용하여 값의 범위를 안정화
        context_risk_feat = math.log1p(weighted_context_risk)

        # 모델에 입력될 최종 딕셔너리 구성
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "time_feats": torch.tensor([last_delta, hour_sin, hour_cos, is_session_start], dtype=torch.float),
            "emo_feats": torch.tensor(emo_vec, dtype=torch.float),
            "context_risk_feats": torch.tensor([context_risk_feat], dtype=torch.float),
        }
        # 레이블이 있는 경우 (학습/검증 데이터)
        if "label" in row.index and not pd.isna(row["label"]):
            item["label"] = torch.tensor(self.label_map.get(row["label"], -1), dtype=torch.long)

        return item

def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """
    DataLoader에서 생성된 샘플 리스트를 미니배치(mini-batch)로 구성합니다.
    가변 길이의 시퀀스(input_ids)를 패딩하여 동일한 길이로 만듭니다.
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    
    # `pad_sequence`를 사용하여 배치 내 최대 길이에 맞춰 패딩을 동적으로 적용
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 나머지 특성들은 텐서로 변환 후 쌓아줍니다 (stack).
    time_feats = torch.stack([b["time_feats"] for b in batch], dim=0)
    emo_feats = torch.stack([b["emo_feats"] for b in batch], dim=0)
    context_risk_feats = torch.stack([b["context_risk_feats"] for b in batch], dim=0)

    out = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "time_feats": time_feats,
        "emo_feats": emo_feats,
        "context_risk_feats": context_risk_feats,
    }
    if "label" in batch[0]:
        out["labels"] = torch.stack([b["label"] for b in batch], dim=0)
    return out

# --- 2. 모델 (Model) ---

class ContextRiskModel(nn.Module):
    """
    문맥을 고려한 위험도 분류 모델.
    사전 학습된 언어 모델(Encoder)과 LSTM, 추가 특성을 결합한 하이브리드 구조.
    """
    def __init__(self, encoder_name: str, emo_feat_dim: int, time_feat_dim: int = 4, num_labels: int = 4, lstm_hidden_size: int = 256, context_risk_feat_dim: int = 1, use_attention: bool = True):
        super().__init__()
        # 모델의 설정을 저장하여 나중에 모델을 불러올 때 동일한 구조를 재현할 수 있도록 함
        self.config = {
            "encoder_name": encoder_name, "emo_feat_dim": emo_feat_dim, "time_feat_dim": time_feat_dim,
            "num_labels": num_labels, "lstm_hidden_size": lstm_hidden_size, 
            "context_risk_feat_dim": context_risk_feat_dim, "use_attention": use_attention,
        }
        self.use_attention = use_attention
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        
        # 양방향 LSTM: 텍스트 시퀀스의 순방향 및 역방향 문맥을 모두 학습
        self.lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        if self.use_attention:
            # Multi-head Attention: LSTM 출력의 여러 부분에 가중치를 부여하여 중요한 정보를 강조
            self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=8, batch_first=True)
            self.attention_norm = nn.LayerNorm(lstm_hidden_size * 2) # 잔차 연결을 위한 Layer Normalization
            pooled_dim = lstm_hidden_size * 2
        else:
            # Attention을 사용하지 않을 경우, LSTM의 마지막 은닉 상태를 사용
            pooled_dim = lstm_hidden_size * 2
        
        # 1. 문맥 위험도를 제외한 특성들로 1차 분류기를 구성합니다.
        # (언어 모델 출력 차원) + (시간 특성 차원) + (감정 특성 차원)
        base_input_dim = pooled_dim + time_feat_dim + emo_feat_dim
        self.base_classifier = nn.Sequential(
            nn.Linear(base_input_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_labels)
        )

        # 2. 문맥 위험도를 '위험도 편향(Risk Bias)'으로 변환하는 작은 네트워크를 추가합니다.
        # 이 네트워크는 positive 점수는 낮추고(-), 나머지 위험도 점수는 높이도록(+) 학습됩니다.
        self.risk_bias_generator = nn.Sequential(
            nn.Linear(context_risk_feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_labels)
        )

    def forward(self, input_ids, attention_mask, time_feats, emo_feats, context_risk_feats):
        # 1. 언어 모델(Encoder)을 통과시켜 토큰별 임베딩(hidden states)을 얻음
        sequence_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 2. LSTM에 입력하기 전, 패딩을 무시하도록 시퀀스를 압축 (성능 및 효율성 향상)
        lengths = attention_mask.sum(dim=1).long().cpu()
        packed_input = pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_out, batch_first=True) # 다시 패딩된 형태로 복원
        
        if self.use_attention:
            # 3a. Attention 적용 및 풀링
            attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output, key_padding_mask=attention_mask == 0)
            # 잔차 연결(Residual Connection) 및 정규화
            pooled = self.attention_norm(lstm_output + attn_output)
            # 어텐션 마스크를 고려하여 평균 풀링 수행
            pooled = self._masked_mean_pooling(pooled, attention_mask)
        else:
            # 3b. Attention 미사용 시, LSTM의 마지막 은닉 상태를 결합하여 사용
            pooled = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 4. 1차 분류: 문맥 위험도를 제외한 특성들로 기본 로짓(logits)을 계산
        base_features = torch.cat([pooled, time_feats, emo_feats], dim=-1)
        base_logits = self.base_classifier(base_features)

        # 5. 위험도 편향(Risk Bias) 계산
        # context_risk_feats가 클수록 이 편향 값의 절대값이 커지도록 학습됩니다.
        risk_bias = self.risk_bias_generator(context_risk_feats)

        # 6. 최종 로짓 = 기본 로짓 + 위험도 편향. 문맥 위험도가 높을수록 위험 클래스 점수가 가산됩니다.
        return base_logits + risk_bias
    
    def _masked_mean_pooling(self, hidden_states, attention_mask):
        """어텐션 마스크를 고려하여 패딩 토큰을 제외하고 평균 풀링을 수행합니다."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9) # 0으로 나누는 것을 방지
        return sum_embeddings / sum_mask

    def save_pretrained(self, save_directory):
        """모델의 가중치와 설정을 저장합니다."""
        os.makedirs(save_directory, exist_ok=True)
        json.dump(self.config, open(os.path.join(save_directory, "config.json"), 'w'), indent=4)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        """저장된 가중치와 설정으로부터 모델을 불러옵니다."""
        config = json.load(open(os.path.join(load_directory, "config.json"), 'r'))
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=torch.device('cpu')))
        return model