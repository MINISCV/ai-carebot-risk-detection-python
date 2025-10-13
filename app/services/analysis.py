import pandas as pd # type: ignore
import torch
import gc
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

    # --- Label Model Prediction ---    
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}

    dataset = ContextDataset(df, label_map)
    batch_items = [dataset[i] for i in range(len(dataset))]
    pad_token_id = label_tokenizer.pad_token_id if label_tokenizer.pad_token_id is not None else 0
    batch = collate_fn(batch_items, pad_token_id)
    inputs = {k: v.to(model_manager.device) for k, v in batch.items()}
    
    with torch.no_grad():
        with autocast(device_type=model_manager.device.type, enabled=(model_manager.device.type == 'cuda')):
            logits = model_manager.label_model(**inputs)
        probabilities = torch.softmax(logits, dim=-1)
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
    del df, encodings, dataset, batch_items, batch, inputs, logits, probabilities, predicted_class_ids, dialogue_result, full_text
    if model_manager.device.type == 'cuda':
        torch.cuda.empty_cache()
    elif model_manager.device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    return result