import base64
from io import BytesIO
from PIL import Image
# Base64로 인코딩된 이미지 데이터
encoded_data = """
"""
# Base64 문자열을 디코딩
image_data = base64.b64decode(encoded_data)
# 바이트 데이터를 이미지로 변환
image = Image.open(BytesIO(image_data))
# 이미지 파일로 저장 (예: 'decoded_image.png')
image.save('decoded_image.png')
# 이미지를 화면에 출력
image.show()