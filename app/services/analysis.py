import pandas as pd # type: ignore
import torch
import gc
import math
from torch.amp import autocast
from typing import List
from fastapi import status
from fastapi.responses import JSONResponse

from app.models.schemas import Dialogue
from app.core.model_loader import ModelManager
from app.models.service_models import ContextDataset, collate_fn
from app.services.preprocessing import (
    parse_datetime_column, compute_time_features, apply_emotional_features, 
    build_context_sequences, clean_text_for_summary
)
from app.core.config import LABEL_ORDER, EVIDENCE_COUNT, MAX_TOTAL_TEXT_LENGTH, K_CONTEXT

def summarize(text: str, model_manager: ModelManager) -> str:
    """
    대화 내용을 요약합니다.
    """
    cleaned_text = clean_text_for_summary(text)
    input_ids = model_manager.summary_tokenizer.encode("summarize: " + cleaned_text, return_tensors="pt").to(model_manager.device)

    with torch.no_grad():
        output_ids = model_manager.summary_model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    return model_manager.summary_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def analyze_dialogues(dialogues: List[Dialogue], model_manager: ModelManager):
    """
    대화 목록을 받아 위험도를 분석하고 요약하는 전체 프로세스를 수행합니다.
    """
    # --- Validation ---
    if not dialogues:
        return JSONResponse(content={"validation_msg": "대화 내용이 비었습니다."}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    total_text_length = sum(len(d.text) for d in dialogues)
    if total_text_length > MAX_TOTAL_TEXT_LENGTH:
        return JSONResponse(content={"validation_msg": f"분석 가능한 최대 글자수를 초과했습니다. ({total_text_length:,} > {MAX_TOTAL_TEXT_LENGTH:,})"}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    if len({d.doll_id for d in dialogues}) != 1:
        return JSONResponse(content={"validation_msg": "모든 대화의 인형 ID는 동일해야 합니다."}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    # --- Label Model Preprocessing ---
    df = pd.DataFrame([d.model_dump() for d in dialogues])
    df = parse_datetime_column(df)
    df = df.sort_values("uttered_at").reset_index(drop=True)
    df = compute_time_features(df)
    df = apply_emotional_features(df)
    df = build_context_sequences(df, k=K_CONTEXT)

    # --- Tokenization for Label Model ---
    label_tokenizer = model_manager.label_tokenizer
    sep = label_tokenizer.sep_token
    df['joined_text'] = [f" {sep} ".join(str(t) for t in texts) for texts in df['seq_texts']]
    
    encodings = label_tokenizer(
        df['joined_text'].tolist(),
        truncation=True, 
        padding=False,
        max_length=label_tokenizer.model_max_length
    )
    df['input_ids'] = encodings['input_ids']
    df['attention_mask'] = encodings['attention_mask']

    # --- Label Model Prediction (Sequential) ---
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    dataset = ContextDataset(df, label_map)

    # 위험도 점수 텐서 (positive:0, danger:1, critical:2, emergency:3)
    risk_scores = torch.tensor([i for i, _ in enumerate(LABEL_ORDER)], dtype=torch.float, device=model_manager.device)
    decay_lambda = 0.00384 # 10분(600초) 지나면 영향력 10%

    # 세션별 과거 위험도 누적을 위한 딕셔너리
    session_context_risks = {}
    all_logits = []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            inputs = {k: v.unsqueeze(0).to(model_manager.device) for k, v in item.items() if k != 'label'}

            # --- 동적 문맥 위험도 계산 ---
            row = dataset.df.iloc[i]
            session_key = (row["doll_id"],) # API에서는 단일 세션으로 간주

            # 세션의 첫 발화인 경우, 문맥 위험도는 0으로 초기화
            if inputs["time_feats"][0, 3].item() == 1.0: # is_session_start
                session_context_risks[session_key] = 0.0
            
            # 현재 발화의 문맥 위험도 특성을 동적으로 계산된 값으로 교체
            current_risk = session_context_risks.get(session_key, 0.0)
            inputs["context_risk_feats"] = torch.tensor([[math.log1p(current_risk)]], dtype=torch.float, device=model_manager.device)
            
            # --- 모델 예측 ---
            with autocast(device_type=model_manager.device.type, enabled=(model_manager.device.type == 'cuda')):
                logits = model_manager.label_model(**inputs)
            all_logits.append(logits)

            # --- 다음 스텝을 위한 문맥 위험도 업데이트 ---
            # 학습 시의 로직과 동일하게, 이전 누적 위험도에 시간 감쇠를 먼저 적용합니다.
            delta_t = torch.exp(inputs["time_feats"][0, 0]) - 1 # log1p 역변환
            decay_factor = math.exp(-decay_lambda * delta_t.item())
            decayed_risk = current_risk * decay_factor

            # 모델의 예측 확률에 위험도 점수를 곱하여 현재 발화의 '기대 위험도'를 계산합니다.
            probs = torch.softmax(logits, dim=-1)
            predicted_risk_score = (probs * risk_scores).sum(dim=-1).item()
            
            # 예측된 위험도 점수가 임계값(e.g., 0.05)보다 클 때만 누적 위험도에 더합니다.
            if predicted_risk_score > 0.05:
                session_context_risks[session_key] = decayed_risk + predicted_risk_score
            else:
                session_context_risks[session_key] = decayed_risk

    all_logits_tensor = torch.cat(all_logits, dim=0)
    probabilities = torch.softmax(all_logits_tensor, dim=-1)
    predicted_class_ids = torch.argmax(probabilities, dim=-1)

    # --- Format Results ---
    dialogue_result = []
    full_text = ""
    for i, row in df.iterrows():
        pred_class_id = predicted_class_ids[i].item()
        label_name = LABEL_ORDER[pred_class_id]
        confidence = {label: f"{score:.4f}" for label, score in zip(LABEL_ORDER, probabilities[i].tolist())}
        
        dialogue_result.append({
            "seq": i,
            "doll_id": row["doll_id"],
            "text": row["text"],
            "uttered_at": row["uttered_at"],
            "label": label_name,
            "confidence_scores": confidence,
        })
        full_text += " " + row["text"]
    
    full_text = full_text.strip()

    # --- Overall Analysis ---
    risk_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    overall_result_item = max(dialogue_result, key=lambda x: risk_map[x['label']])
    overall_label_name = overall_result_item['label']
    overall_confidence = overall_result_item['confidence_scores']

    # --- Treatment Plan ---
    treatment_plan_map = {
        "positive": "특별한 위험 징후는 없습니다. 지속적으로 모니터링해 주세요.",
        "danger": "주의가 필요한 발화가 감지되었습니다. 반복될 경우 주기적인 안부 확인 및 말벗 서비스 제공을 권장합니다.",
        "critical": "위험도가 높은 발화가 감지되었습니다. 상황에 따라 관리자가 직접 통화하여 심리적 안정을 유도하고, 방문 상담이 필요할 수 있습니다.",
        "emergency": "매우 위급한 발화가 감지되었습니다. 신속하게 상황을 파악한 후 관계 기관에 신고하거나 적극적인 대응이 요구됩니다."
    }
    treatment_plan = treatment_plan_map.get(overall_label_name, "잘못된 위험도 분류입니다.")

    summary_text = summarize(full_text, model_manager)
    evidences = sorted(dialogue_result, key=lambda x: float(x["confidence_scores"][overall_label_name]), reverse=True)
    evidences = [{"seq": v["seq"], "text": v["text"], "score": v["confidence_scores"][overall_label_name]} for v in evidences][:EVIDENCE_COUNT]

    result = {
        "overall_result": {
            "doll_id": dialogues[0].doll_id,
            "dialogue_count": len(dialogues),
            "char_length": len(full_text),
            "label": overall_label_name,
            "confidence_scores": overall_confidence,
            "treatment_plan": treatment_plan,
            "full_text": full_text,
            "reason": {
                "evidence": evidences,
                "summary": summary_text
            },
        },
        "dialogue_result": dialogue_result,
    }

    # --- Memory Cleanup ---
    del df, encodings, dataset, inputs, all_logits, all_logits_tensor, probabilities, predicted_class_ids, dialogue_result, full_text
    if model_manager.device.type == 'cuda':
        torch.cuda.empty_cache()
    elif model_manager.device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return result