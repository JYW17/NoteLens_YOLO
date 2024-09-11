from fastapi import APIRouter, File, UploadFile, HTTPException
import requests
import os
from io import BytesIO
from pathlib import Path
import logging
from app.service.yoloService import YOLOv5Service

# 로그 설정
logging.config.fileConfig('app/config/logging_config.ini')
logger = logging.getLogger(__name__)

# 윈도우에서만 실행할 코드 - PosixPath를 WindowsPath로 변경
if os.name == 'nt':
    from pathlib import Path
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
else:
    from pathlib import Path

# YOLOv5 서비스 인스턴스 생성 (사용자 지정 가중치 경로를 전달)
yolov5_service = YOLOv5Service()

yoloRouter = APIRouter()

# SERVER2_HEALTH_URL = "http://localhost:8000/api/test/health"  # 로컬 테스트용 주소
# SERVER2_OCR_MULTI_URL = "http://localhost:8000/api/ocr/ocr-multi"  # 로컬 테스트용 주소

# 서버2의 주소 및 OCR 서비스 주소
SERVER2_HEALTH_URL = "http://43.203.54.176:8000/api/test/health"
SERVER2_OCR_MULTI_URL = "http://43.203.54.176:8000/api/ocr/ocr-multi"  # 이민섭 ocr서버 api 주소
# SERVER2_OCR_MULTI_URL = "http://43.203.93.209:8000/api/ocr/ocr-multi"  # 주영운 ec2의 ocr서버 api 주소

# 식별자
class Temp_id:
    def __init__(self):
        self.id = 0
    def get_id(self):
        self.id += 1
        return f"No_{self.id}_file"

temp_id = Temp_id()

# 파일을 직접 받아서 작업하는 API
@yoloRouter.post("/yolo", response_model=dict)
async def process_image(file: UploadFile = File(...)):
    
    # 서버2가 정상적으로 작동하는지 확인
    if not await yolov5_service.is_server2_healthy(SERVER2_HEALTH_URL):
        logger.error("Server2 is not healthy")
        raise HTTPException(status_code=500, detail="Server2 is not healthy")
    
    # 임시 저장할 파일 경로
    file_id = temp_id.get_id()
    temp_file_path = await yolov5_service.save_temp_file(file, file_id)
    logger.info(f"임시 파일 경로: {temp_file_path}")
    
    # yolo로 이미지 크롭 수행
    textDetectionResult = yolov5_service.textDetection(temp_file_path, file_id)
    logger.info(f"yolo로 이미지 크롭 수행 결과: {textDetectionResult}")
    
    # 임시 파일 삭제
    os.remove(temp_file_path)
    logger.info("임시 파일을 성공적으로 삭제했습니다")

    # 크롭된 이미지들을 OCR 서버로 전송, 파일 삭제, 결과 반환
    dir_path = Path("yolov5") / "runs" / "detect" / file_id / "crops" # 크롭된 이미지 경로
    remove_folder_path = Path("yolov5") / "runs" / "detect" / file_id
    return await yolov5_service.send_cropped_images_to_ocr(dir_path, remove_folder_path, SERVER2_OCR_MULTI_URL)
    # return {"message": "success"}


# URL로 이미지를 받아서 작업하는 API
@yoloRouter.post("/yolo-from-url", response_model=dict)
async def process_image_from_url(image_url: str):
    # 서버2가 정상적으로 작동하는지 확인
    if not await yolov5_service.is_server2_healthy(SERVER2_HEALTH_URL):
        logger.error("Server2 is not healthy")
        raise HTTPException(status_code=500, detail="Server2 is not healthy")

    # 몽고아이디
    file_id = temp_id.get_id()

    # 이미지를 받아서 BytesIO 객체로 변환
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_bytes = BytesIO(response.content)
    
    # 임시 저장할 파일 경로
    temp_image_path = f"temp_{image_url.split('/')[-1]}"
    logger.info(f"임시 파일 경로: {temp_image_path}")

    with open(temp_image_path, 'wb') as image_file:
        image_file.write(image_bytes.read())
    
    # yolo로 이미지 크롭 수행
    yolov5_service.textDetection(temp_image_path, file_id)
    
    # 임시 파일 삭제
    os.remove(temp_image_path)
    logger.info("임시 파일을 성공적으로 삭제했습니다 - 욜로 크롭 수행 종료")

    # 크롭된 이미지들을 OCR 서버로 전송
    dir_path = Path("yolov5") / "runs" / "detect" / file_id / "crops" # 크롭된 이미지 경로
    remove_folder_path = Path("yolov5") / "runs" / "detect" / file_id
    return await yolov5_service.send_cropped_images_to_ocr(dir_path, remove_folder_path, SERVER2_OCR_MULTI_URL)