import os
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from transformers import AutoModel
from typing import List, Dict, Any

# --- 1. 데이터셋 클래스 및 함수 (Dataset Classes and Functions) ---

class ContextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: Dict[str, int]):
        self.df = df
        self.label_map = label_map
        self.emo_cols = [c for c in df.columns if c.startswith("emo_")]
        self.emo_score_indices = {
            'emergency': self.emo_cols.index('emo_emergency_score') if 'emo_emergency_score' in self.emo_cols else None,
            'critical': self.emo_cols.index('emo_critical_score') if 'emo_critical_score' in self.emo_cols else None,
            'danger': self.emo_cols.index('emo_danger_score') if 'emo_danger_score' in self.emo_cols else None,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]

        last_delta = row["delta_t"]
        last_hour = row["hour"]
        hour_sin = math.sin(2 * math.pi * last_hour / 24)
        hour_cos = math.cos(2 * math.pi * last_hour / 24)
        emo_vec = row[self.emo_cols].values.astype(np.float32)

        seq_emo_vectors = row["seq_emo_vectors"]
        seq_delta_t = row["seq_delta_t"]
        
        weighted_context_risk = 0.0
        if len(seq_emo_vectors) > 1:
            for i in range(len(seq_emo_vectors) - 1):
                emo_vec_context = seq_emo_vectors[i]
                delta_t = seq_delta_t[i+1]
                utterance_risk_score = 0
                if self.emo_score_indices['emergency'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['emergency']] * 3.0
                if self.emo_score_indices['critical'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['critical']] * 2.0
                if self.emo_score_indices['danger'] is not None: utterance_risk_score += emo_vec_context[self.emo_score_indices['danger']] * 1.0
                if utterance_risk_score > 0: weighted_context_risk += utterance_risk_score / (delta_t + 1.0)
        
        context_risk_feat = math.log1p(weighted_context_risk)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "time_feats": torch.tensor([last_delta, hour_sin, hour_cos], dtype=torch.float),
            "emo_feats": torch.tensor(emo_vec, dtype=torch.float),
            "context_risk_feats": torch.tensor([context_risk_feat], dtype=torch.float),
        }
        if "label" in row.index and not pd.isna(row["label"]):
            item["label"] = torch.tensor(self.label_map.get(row["label"], -1), dtype=torch.long)

        return item

def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

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
    def __init__(self, encoder_name: str, emo_feat_dim: int, time_feat_dim: int = 3, num_labels: int = 4, lstm_hidden_size: int = 256, context_risk_feat_dim: int = 1, use_attention: bool = True):
        super().__init__()
        self.config = {
            "encoder_name": encoder_name, "emo_feat_dim": emo_feat_dim, "time_feat_dim": time_feat_dim,
            "num_labels": num_labels, "lstm_hidden_size": lstm_hidden_size, 
            "context_risk_feat_dim": context_risk_feat_dim, "use_attention": use_attention,
        }
        self.use_attention = use_attention
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size
        
        self.lstm = nn.LSTM(input_size=enc_dim, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=8, batch_first=True)
            self.attention_norm = nn.LayerNorm(lstm_hidden_size * 2)
            pooled_dim = lstm_hidden_size * 2
        else:
            pooled_dim = lstm_hidden_size * 2
            
        input_dim = pooled_dim + time_feat_dim + emo_feat_dim + context_risk_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, time_feats, emo_feats, context_risk_feats):
        sequence_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).long().cpu()
        packed_input = pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        if self.use_attention:
            attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output, key_padding_mask=attention_mask == 0)
            pooled = self.attention_norm(lstm_output + attn_output)
            pooled = self._masked_mean_pooling(pooled, attention_mask)
        else:
            pooled = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        x = torch.cat([pooled, time_feats, emo_feats, context_risk_feats], dim=-1)
        return self.classifier(x)
    
    def _masked_mean_pooling(self, hidden_states, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        json.dump(self.config, open(os.path.join(save_directory, "config.json"), 'w'), indent=4)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, load_directory):
        config = json.load(open(os.path.join(load_directory, "config.json"), 'r'))
        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=torch.device('cpu')))
        return model