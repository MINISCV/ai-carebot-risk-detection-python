# 고독사 예방을 위한 시니어케어 돌봄로봇 데이터 분석 - python

## 요구 사항 및 설치

### 파이선 버전 및 주요 의존성

-   python == 3.13
-   fastapi == 0.118.3
-   uvicorn == 0.37.0
-   torch == 2.8.0
-   pandas == 2.3.3
-   numpy == 2.1.2
-   transformers == 4.57.0

### 패키지 설치

패키지를 직접 설치할 경우

```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 # 개발 환경에 따라 주의
pip install fastapi uvicorn transformers pandas
```

또는, `requirements.txt` 파일을 이용할 경우

```bash
pip install -r requirements.txt
```

### 모델 준비

학습이 완료된 모델 파일은 [시니어 발화 데이터 분석을 통한 위험도 분류 및 대화 요약 모델](https://github.com/odigano/pkdt-ai_carebot_risk_detection_data_kisti) 프로젝트의 `README.md` 파일을 참고해주세요.

위험도 분류 모델은 **klue/roberta-base** 사전 학습 모델을 기반으로 하였으며, `model/label` 폴더로 모델 파일을 이동시켜주세요.

요약 모델은 **paust/pko-t5-small** 사전 학습 모델을 기반으로 하였으며, `model/summary` 폴더로 모델 파일을 이동시켜주세요.

## 실행

```bash
uvicorn main:app --reload # 기본 주소 - http://127.0.0.1:8000
```

## 대화 분석 요청 API

### 요청 예시

```
POST /analyze HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Content-Length: 335

[
    {
        "doll_id": "1",
        "text": "오늘 너무 덥네",
        "uttered_at": "2025-09-22 10:20:30"
    },
    {
        "doll_id": "1",
        "text": "지금 몇 시야",
        "uttered_at": "2025-09-22 10:20:40"
    },
    {
        "doll_id": "1",
        "text": "조금 있다가 밥 먹어야 겠다",
        "uttered_at": "2025-09-22 10:20:50"
    }
]
```

### 응답 예시 - 정상

```json
// 200 OK
{
    "overall_result": {
        "doll_id": "1",
        "dialogue_count": 3,
        "char_length": 32,
        "label": "positive",
        "confidence_scores": {
            "positive": "0.9907",
            "danger": "0.0056",
            "critical": "0.0029",
            "emergency": "0.0010"
        },
        "treatment_plan": "특별한 위험 징후는 없습니다. 지속적으로 모니터링해 주세요.",
        "full_text": "오늘 너무 덥네 지금 몇 시야 조금 있다가 밥 먹어야 겠다",
        "reason": {
            "evidence": [
                {
                    "seq": 0,
                    "text": "오늘 너무 덥네",
                    "score": "0.9907"
                },
                {
                    "seq": 1,
                    "text": "지금 몇 시야",
                    "score": "0.9902"
                }
            ],
            "summary": "오늘 너무 덥다고 말하며 밥을 먹어야겠다고 함"
        }
    },
    "dialogue_result": [
        {
            "seq": 0,
            "doll_id": "1",
            "text": "오늘 너무 덥네",
            "uttered_at": "2025-09-22T10:20:30",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9907",
                "danger": "0.0056",
                "critical": "0.0029",
                "emergency": "0.0010"
            }
        },
        {
            "seq": 1,
            "doll_id": "1",
            "text": "지금 몇 시야",
            "uttered_at": "2025-09-22T10:20:40",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9902",
                "danger": "0.0027",
                "critical": "0.0055",
                "emergency": "0.0018"
            }
        },
        {
            "seq": 2,
            "doll_id": "1",
            "text": "조금 있다가 밥 먹어야 겠다",
            "uttered_at": "2025-09-22T10:20:50",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9897",
                "danger": "0.0041",
                "critical": "0.0050",
                "emergency": "0.0011"
            }
        }
    ]
}
```

### 응답 예시 - 유효성 검사 실패 (요청 본문의 대화 로그 리스트가 비어있음)

```json
// 422 Unprocessable Content
{
    "validation_msg": "empty_list"
}
```

### 응답 예시 - 유효성 검사 실패 (요청 본문 최대 문자수 제한 초과)

```json
// 422 Unprocessable Content
{
    "validation_msg": "char_limit_over (13498 > 10000)"
}
```

### 응답 예시 - 유효성 검사 실패 (대화 로그 리스트 중 유니크한 doll_id 값이 1개가 아님)

```json
// 422 Unprocessable Content
{
    "validation_msg": "invalid_doll_id"
}
```

### 응답 예시 - 유효성 검사 실패 (pydantic 라이브러리 검증 실패 포맷)

```json
// 422 Unprocessable Content
{
    "detail": [
        {
            "type": "missing",
            "loc": ["body", 0, "doll_id"],
            "msg": "Field required",
            "input": {
                "text": "오늘 너무 덥네",
                "uttered_at": "2025-09-22 10:20:30"
            }
        }
    ]
}
```
