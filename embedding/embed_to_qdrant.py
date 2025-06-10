import json
import hashlib
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# 1. JSON 파일 경로
file_path = "embedding/data_with_predicted_final_tagged.json"

# 2. JSON 데이터 로딩
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 임베딩 모델 로딩
print("✅ KoSimCSE 모델 로딩 중...")
model = SentenceTransformer("BM-K/KoSimCSE-roberta")

# 4. Qdrant Cloud 연결
client = QdrantClient(
    url="https://2a09054d-de92-436e-bf8c-158f44d82df4.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1VAec5LQRLXCskcPERZg3WgNpTpj00q4ZwVqVkCy0RA",
    timeout=60.0
)

# 5. 예측 카테고리 기준으로 분리
category_map = {}
for item in data:
    big_category = item.get("predicted_category") 
    if big_category not in category_map:
        category_map[big_category] = []
    category_map[big_category].append(item)

# 6. 카테고리별 저장
for big_category, items in category_map.items():
    collection_name = f"dku_{big_category}"
    print(f"\n📌 {collection_name} → {len(items)}개 저장 중...")

    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    points = []
    for item in tqdm(items, desc=f"{collection_name} 임베딩"):
        question = item.get("question", "")
        answer = item.get("answer", "")
        vector = model.encode(question).tolist()

        uid = item.get("id")
        if not isinstance(uid, int):
            uid = int(hashlib.sha256((question + answer).encode()).hexdigest(), 16) % (10**9)

        point = PointStruct(
            id=uid,
            vector=vector,
            payload=item
        )
        points.append(point)

    print("📤 Qdrant에 배치 업로드 중...")
    for i in tqdm(range(0, len(points), 100), desc="업로드 진행"):
        batch = points[i: i + 100]
        for retry in range(3):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                break
            except Exception as e:
                print(f"업로드 실패 (재시도 {retry+1}/3): {e}")
                time.sleep(5)
        else:
            print("❌ 실패한 배치가 존재합니다.")

print("\n✅ 모든 카테고리 저장 완료!")
