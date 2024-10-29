# 실행 명령어 uvicorn main:app --reload --port=8001
# 포트번호 : 8000번
# api list : localhost:8001/docs
from fastapi import FastAPI
from app.router import imageRouter, yoloRouter

import logging.config
from app.service.yoloService import YOLOv5Service
import os
import shutil
import time

# 윈도우에서만 실행할 코드 - PosixPath를 WindowsPath로 변경
if os.name == 'nt':
    from pathlib import Path
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
else:
    from pathlib import Path

app = FastAPI()


# 로그 설정
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    
    # 로그 초기화
    # initialize_log()
    
    # 모델 초기화
    initialize_models()
    
    # 데이터베이스 초기화
    initialize_database()
    
    logger.info("서버가 초기화되었습니다.")


def initialize_models():
    # Yolo 모델 초기화 코드 구현
    yolov5_service = YOLOv5Service()
    test_image_path = Path("yolov5") / "data" / "images" / "bus.jpg" # 테스트 이미지 경로
    try:
        start_time = time.time()
        yolov5_service.textDetection(image_path=test_image_path, file_id="test_image", save_csv=False, save_txt=False, save_crop=False)
        end_time = time.time()
        logger.info(f"YOLO로 테스트 이미지 {test_image_path} 처리 완료 (소요시간: {end_time - start_time:.2f}초)")
    except Exception as e:
        logger.error(f"테스트 이미지 처리 중 오류 발생: {e}")
    
    logger.info("모델이 초기화되었습니다.")

def initialize_log():
    # 로그 파일 초기화 코드 구현
    log_file = Path("app.log")
    try:
        # 로그 파일 속 내용 삭제
        with open(log_file, "w"):
            pass
    except Exception as e:
        logger.error(f"로그 파일 삭제 중 오류 발생: {e}")
    logger.info("로그 파일이 초기화되었습니다.")

def initialize_database():
    
    # 데이터베이스 초기화 코드 구현
    runs_dir = "yolov5/runs/detect"
    temp_dir = "temp_images"
    directories_to_clear = [runs_dir, temp_dir]

    for directory in directories_to_clear:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
            logger.info(f"'{directory}' 디렉토리가 비워졌습니다.")
        else:
            os.makedirs(directory)
            logger.warning(f"'{directory}' 디렉토리가 존재하지 않아 새로 생성되었습니다.")
    logger.info("데이터베이스가 초기화되었습니다.")


# 라우터 정의
app.include_router(imageRouter, prefix="/api/image")
app.include_router(yoloRouter, prefix="/api/yolo")
