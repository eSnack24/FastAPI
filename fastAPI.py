from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
from starlette.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
import requests

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        return embeddings.squeeze().numpy()
    except UnidentifiedImageError:
        print(f"Error: {image_path} is not a valid image.")
        return None

# 이미지 폴더 경로
image_folder = "C:\\zzz\\snack"
app.mount("/static", StaticFiles(directory=image_folder), name="static")

# 이미지 폴더에서 모든 이미지 임베딩을 ChromaDB에 저장
def load_image_embeddings():
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            embedding = get_image_embedding(image_path)
            if embedding is not None:
                collection.add(
                    ids=[image_file],  # 이미지 파일명을 고유 ID로 사용
                    documents=[image_file],  # 문서로 이미지 파일명 추가
                    embeddings=[embedding]  # 임베딩 추가
                )

# 이미지 임베딩을 로드
load_image_embeddings()

SPRING_SERVER_URL = "http://localhost:8080/api/v1/saveState"

# 유사 이미지 검색
@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    try:
        # 업로드된 이미지에서 임베딩 생성
        image = Image.open(file.file)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            new_embedding = model.get_image_features(**inputs).squeeze().numpy()

        # ChromaDB에서 유사 이미지 검색
        results = collection.query(query_embeddings=[new_embedding], n_results=5)
        similar_images = results["documents"]

        # 유사 이미지 검색 결과를 Spring 서버로 보내기
        payload = {"similar_images": similar_images}
        headers = {"Content-Type": "application/json"}
        response = requests.post(SPRING_SERVER_URL, json=payload, headers=headers)

        # Spring 서버의 응답 처리
        if response.status_code == 200:
            return JSONResponse(content={"similar_images": similar_images, "spring_response": response.json()})
        else:
            return JSONResponse(content={"similar_images": similar_images, "spring_error": "Failed to communicate with Spring server"})

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 파일 업로드
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일 저장 경로 생성
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)

        # 파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse(content={"message": "File uploaded successfully", "filename": file.filename})

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

# FastAPI 애플리케이션 실행 (Uvicorn)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
