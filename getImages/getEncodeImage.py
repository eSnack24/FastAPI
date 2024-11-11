from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import time

app = FastAPI()

# 이미지가 저장될 폴더 경로
UPLOAD_FOLDER = "C:\\zzz\\snack"

# 폴더가 없다면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class ImageData(BaseModel):
    image_data: str

@app.post("/upload_image")
async def upload_image(data: ImageData):
    try:
        base64_image = data.image_data

        # Base64 디코딩
        image_data = base64.b64decode(base64_image)

        # 타임스탬프를 파일 이름에 추가 (예: 20231110123045.png)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        image_filename = f"received_image_{timestamp}.png"

        # 파일 경로 지정
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        # 파일로 저장
        with open(image_path, "wb") as file:
            file.write(image_data)

        return {"message": f"Image received and saved at {image_path}"}

    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to process the image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
