# 고독사 예방을 위한 시니어케어 돌봄로봇 데이터 분석 - python

## 파이선 버전
- python 3.13

## 패키지 설치
```bash
# 주요 패키지 직접 설치할 경우
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # 개발 환경에 따라 주의
pip install fastapi uvicorn transformers pandas scikit-learn 

# requirements.txt 파일을 이용할 경우
pip install -r requirements.txt
```

## 서버 실행
```bash
# 기본 주소 - http://127.0.0.1:8000
uvicorn main:app
uvicorn main:app --reload # 서버 실행 중 파일 갱신 시 자동으로 서버 재시작
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
            "positive": "0.9995",
            "danger": "0.0002",
            "critical": "0.0001",
            "emergency": "0.0000"
        },
        "full_text": "오늘 너무 덥네 지금 몇 시야 조금 있다가 밥 먹어야 겠다",
        "reason": {
            "evidence": [
                {
                    "seq": 1,
                    "text": "지금 몇 시야",
                    "score": "1.0000"
                },
                {
                    "seq": 0,
                    "text": "오늘 너무 덥네",
                    "score": "0.9995"
                }
            ],
            "summary": "오늘 너무 덥다고 말하며 조금 있다가 밥을 먹어야겠다고 함"
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
                "positive": "0.9995",
                "danger": "0.0002",
                "critical": "0.0001",
                "emergency": "0.0000"
            }
        },
        {
            "seq": 1,
            "doll_id": "1",
            "text": "지금 몇 시야",
            "uttered_at": "2025-09-22T10:20:40",
            "label": "positive",
            "confidence_scores": {
                "positive": "1.0000",
                "danger": "0.0001",
                "critical": "0.0001",
                "emergency": "0.0001"
            }
        },
        {
            "seq": 2,
            "doll_id": "1",
            "text": "조금 있다가 밥 먹어야 겠다",
            "uttered_at": "2025-09-22T10:20:50",
            "label": "positive",
            "confidence_scores": {
                "positive": "0.9995",
                "danger": "0.0002",
                "critical": "0.0001",
                "emergency": "0.0000"
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
### 응답 예시 - 유효성 검사 실패 (대화 로그 리스트 중 유니크한 doll_id 값이 1개가 아님)
```json
// 422 Unprocessable Content
{
    "validation_msg": "invalid_doll_id"
}
```
### 응답 예시 - 유효성 검사 실패 (pydantic 라이브러리 검증 실패 포맷 / `TODO` 포맷 커스텀 처리 검토)
```json
// 422 Unprocessable Content
{
    "detail": [
        {
            "type": "missing",
            "loc": [
                "body",
                0,
                "doll_id"
            ],
            "msg": "Field required",
            "input": {
                "text": "오늘 너무 덥네",
                "uttered_at": "2025-09-22 10:20:30"
            }
        }
    ]
}
```
