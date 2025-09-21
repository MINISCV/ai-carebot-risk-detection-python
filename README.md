# 고독사 예방을 위한 시니어케어 돌봄로봇 데이터 분석 - python

## 파이선 버전
- python 3.13

## 패키지 설치
```bash
# 주요 패키지 직접 설치할 경우
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # 개발 환경에 따라 주의
pip install fastapi uvicorn transformers

# requirements.txt 파일을 이용할 경우
pip install -r requirements.txt
```

## 서버 실행
```bash
# 기본 주소 - http://127.0.0.1:8000
uvicorn main:app
```

## 대화 분석 요청 API
### 요청 예시
```
POST /analyze HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Content-Length: 329

[
    {
        "dollId": "1",
        "text": "오늘 너무 덥네",
        "utteredAt": "2025-09-22 10:20:30"
    },
    {
        "dollId": "1",
        "text": "지금 몇 시야",
        "utteredAt": "2025-09-22 10:20:40"
    },
    {
        "dollId": "1",
        "text": "조금 있다가 밥 먹어야 겠다",
        "utteredAt": "2025-09-22 10:20:50"
    }
]
```
### 응답 예시
```json

{
    "result": "success",
    "validation_msg": "",
    "overall_result": {
        "doll_id": "1",
        "dialogue_count": 3,
        "char_length": 32,
        "label": "positive",
        "confidence_scores": {
            "positive": "0.9962",
            "danger": "0.0031",
            "critical": "0.0004",
            "emergency": "0.0003"
        },
        "full_text": "오늘 너무 덥네 지금 몇 시야 조금 있다가 밥 먹어야 겠다",
        "reason": {
            "evidence": [
                {
                    "seq": 1,
                    "text": "지금 몇 시야",
                    "score": "0.9984"
                },
                {
                    "seq": 0,
                    "text": "오늘 너무 덥네",
                    "score": "0.9969"
                }
            ],
            "summary": "오늘 너무 덥다고 말하며, 밥 먹어야겠다고 함. 식사에 대해 이야기함"
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
                "positive": "0.9969",
                "danger": "0.0025",
                "critical": "0.0003",
                "emergency": "0.0004"
            }
        },
        {
            "seq": 1,
            "doll_id": "1",
            "text": "지금 몇 시야",
            "uttered_at": "2025-09-22T10:20:40",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9984",
                "danger": "0.0011",
                "critical": "0.0002",
                "emergency": "0.0003"
            }
        },
        {
            "seq": 2,
            "doll_id": "1",
            "text": "조금 있다가 밥 먹어야 겠다",
            "uttered_at": "2025-09-22T10:20:50",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9967",
                "danger": "0.0027",
                "critical": "0.0004",
                "emergency": "0.0002"
            }
        }
    ]
}
```
