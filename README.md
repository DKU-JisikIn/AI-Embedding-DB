# AI-Embedding-DB


> 질문/답변 데이터를 카테고리 기반으로 임베딩하고 Qdrant 벡터 DB에 저장하는 시스템

---

## 📌 프로젝트 개요

단국대학교 VOC에서 수집된 질문-답변 데이터를 KoSimCSE 모델로 임베딩하여, **Qdrant** 벡터 데이터베이스에 저장하기 위한 Python 스크립트를 포함합니다.

- 질문-답변 데이터를 **카테고리별로 분류**하여 Qdrant의 컬렉션으로 나눠 저장
- 추후 RAG 기반 응답 생성을 위한 **벡터 검색 인프라 구성**
- 분류된 카테고리는 모델 추론 결과(`predicted_category`) 또는 기존 메타데이터(`category`)를 기반으로 설정 가능

---

## 🧩 주요 기능

| 기능 | 설명 |
|------|------|
| ✅ JSON 데이터 로딩 | 크롤링 또는 분류된 `.json` 파일을 불러옴 |
| ✅ KoSimCSE 모델 임베딩 | 질문 텍스트를 임베딩 (768-dimension) |
| ✅ Qdrant Cloud 연결 | API Key 기반 원격 Qdrant 서버 연결 |
| ✅ 카테고리별 벡터 DB 저장 | predicted_category 기준으로 컬렉션 생성 및 저장 |
| ✅ 배치 업로드 + 재시도 | 네트워크 타임아웃 방지를 위한 100개 단위 업로드, 실패 시 재시도 |

---

## 🧱 프로젝트 구조

```
AI-Embedding-DB/
├── embedding/
│   ├── data_with_predicted_final_tagged.json  # 분류 모델 결과 포함 데이터
│   └── embed_to_qdrant.py          # 메인 임베딩 + 저장 스크립트
├── README.md
└── requirements.txt
```


---

## ⚙️ 실행 환경

- Python 3.10+
- `sentence-transformers`
- `qdrant-client`
- `tqdm`

설치:
```bash
pip install -r requirements.txt

---

🚀 실행 방법
python embedding/embed_to_qdrant.py

---

🔒 Qdrant Cloud 설정 방법
https://cloud.qdrant.io 접속

무료 계정으로 클러스터 생성 (Starter 요금제 무료)

Cluster URL + API Key 발급

embed_to_qdrant.py의 QdrantClient 설정에 적용

---

🧪 예시
dku_일반, dku_교양, dku_장학, ... 등의 컬렉션이 Qdrant에 생성되어 저장됩니다.

각 컬렉션에 최대 수천 개의 질문 벡터가 저장되어 검색 가능한 상태가 됩니다.

---

📬 Contact
Maintainer: 서민경

School: 단국대학교

프로젝트 관련 문의는 Issues 또는 이메일로 주세요.



