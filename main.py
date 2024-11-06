from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from starlette.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ChromaDB 클라이언트 설정
client = chromadb.Client()
collection = client.create_collection("image_embeddings")

# 이미지 파일에서 임베딩 생성
def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy()

# 이미지 폴더 경로
image_folder = "C:\\zzz\\snack"
app.mount("/static", StaticFiles(directory=image_folder), name="static")

# 이미지 폴더에서 모든 이미지 임베딩을 ChromaDB에 저장
def load_image_embeddings():
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        embedding = get_image_embedding(image_path)

        collection.add(
            ids=[image_file],  # 이미지 파일명을 고유 ID로 사용
            documents=[image_file],  # 문서로 이미지 파일명 추가
            embeddings=[embedding]  # 임베딩 추가
        )

# 이미지 임베딩을 로드
load_image_embeddings()

# 유사 이미지 검색 엔드포인트
@app.post("/search/")
async def search_similar_images(file: UploadFile = File(...)):
    # 업로드된 이미지에서 임베딩 생성
    image = Image.open(file.file)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        new_embedding = model.get_image_features(**inputs).squeeze().numpy()

    # ChromaDB에서 유사 이미지 검색
    results = collection.query(query_embeddings=[new_embedding], n_results=2)

    # 유사 이미지 파일명 반환
    return JSONResponse(content={"similar_images": results["documents"]})

# FastAPI 애플리케이션 실행 (Uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
