import logging
import logging.config
import os
import shutil
from pathlib import Path
import httpx
from fastapi import HTTPException
from yolov5 import detection

# 로그 설정
logging.config.fileConfig('app/config/logging_config.ini')

class YOLOv5Service:
    def __init__(self):
        self.detection = detection.run
        self.logger = logging.getLogger(__name__)
        self.logger.info("YOLOv5Service 인스턴스 생성됨")

    def textDetection(self, image_path, mongo_id):
        self.logger.info(f"textDetection 함수 실행 - 이미지 경로: {image_path}, 몽고 아이디: {mongo_id}")
        # 크롭된 이미지들이 경로에 저장됨
        self.detection(source=image_path, mongo_id=mongo_id)

    async def is_server2_healthy(self, health_url):
        self.logger.info(f"서버 상태 확인 - URL: {health_url}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(health_url, timeout=5)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    self.logger.info("서버2 상태: 정상")
                    return True
            except httpx.RequestError:
                self.logger.error("서버2 상태: 비정상")
                return False
        return False

    async def save_temp_file(self, file) -> str:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        self.logger.info(f"임시 파일 저장 - 경로: {temp_file_path}")
        return temp_file_path

    async def send_cropped_images_to_ocr(self, dir_path: Path, remove_folder_path: Path, ocr_url: str) -> dict:
        self.logger.info(f"OCR 서버로 크롭된 이미지 전송 - 디렉토리 경로: {dir_path}, 제거할 폴더 경로: {remove_folder_path}, OCR URL: {ocr_url}")
        files_data = []
        open_files = []
        for file_path in dir_path.glob("*.jpg"):
            f = open(file_path, "rb")
            files_data.append(('files', (file_path.name, f, 'image/jpeg')))
            open_files.append(f)

        if not files_data:
            for f in open_files:
                f.close()
            self.logger.info("모든 파일 객체를 닫았습니다.")
            if os.path.exists(remove_folder_path):
                shutil.rmtree(remove_folder_path)
                self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
            else:
                self.logger.info(f"폴더 '{remove_folder_path}' 가 존재하지 않습니다.")
            self.logger.error("크롭된 이미지 파일이 존재하지 않습니다.")
            raise HTTPException(status_code=404, detail="크롭된 이미지 파일이 존재하지 않습니다.")

        self.logger.info(f"httpx 작업 전 - 수신지 url: {ocr_url}")
        async with httpx.AsyncClient() as client:
            try:
                timeout_limit = 45
                response = await client.post(url=ocr_url, files=files_data, timeout=timeout_limit)
                response.raise_for_status()
                result_texts = response.json()
                self.logger.info("httpx를 통해 ocr 서버의 api로부터 리턴값 받음")
                
                # OCR 결과값들이 3, 4, 2, 5, 1... 등의 순서로 오는 경우가 있어서 정렬
                sorted_keys = sorted(result_texts.keys())
                sorted_data = {key: result_texts[key] for key in sorted_keys}
                self.logger.info(f"정렬된 json 데이터를 반환합니다.{sorted_data}")
                return sorted_data

            except httpx.TimeoutException:
                self.logger.error("Request timeout while requesting server2")
                raise HTTPException(status_code=504, detail=f"Server2 did not respond in time. timeout limit is {timeout_limit} seconds.")
            except httpx.RequestError as exc:
                self.logger.error(f"Request error while requesting server2: {exc}")
                raise HTTPException(status_code=500, detail=f"Error while requesting server2: {exc}")
            except httpx.HTTPStatusError as exc:
                self.logger.error(f"HTTP error response from server2: {exc.response.text}")
                raise HTTPException(status_code=exc.response.status_code, detail=f"Error response from server2: {exc.response.text}")
            except Exception as exc:
                self.logger.error(f"Unexpected error: {exc}")
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {exc}")
            finally:
                for f in open_files:
                    f.close()
                self.logger.info("모든 파일 객체를 닫았습니다.")
                if os.path.exists(remove_folder_path):
                    shutil.rmtree(remove_folder_path)
                    self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
                else:
                    self.logger.info(f"폴더 '{remove_folder_path}' 가 존재하지 않습니다.")