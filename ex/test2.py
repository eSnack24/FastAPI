from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

import chromadb

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 파일을 로드하고 임베딩 생성
def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy()


client = chromadb.Client()
collection = client.create_collection("image_embeddings")

# 이미지 임베딩을 ChromaDB에 저장
import os

image_folder = "C:\\zzz\\snack"
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    embedding = get_image_embedding(image_path)

    print(image_file)

    collection.add(
        ids=[image_file],  # 이미지 파일명을 고유 ID로 사용
        documents=[image_file],  # 문서로 이미지 파일명 추가
        embeddings=[embedding]  # 임베딩 추가
    )


new_image_path = "C:\\zzz\\라면.jpeg"
new_embedding = get_image_embedding(new_image_path)

print("=========================================================")

# ChromaDB에서 유사 이미지 검색
results = collection.query(query_embeddings=[new_embedding], n_results=5)  # top_k는 검색 결과 개수
for result in results["documents"]:
    print(result)  # 유사 이미지 파일명 출력