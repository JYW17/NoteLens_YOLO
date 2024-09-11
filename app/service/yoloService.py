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

    def textDetection(self, image_path, file_id):
        self.logger.info(f"textDetection 함수 실행 - 이미지 경로: {image_path}, 파일 아이디: {file_id}")
        # 크롭된 이미지들이 경로에 저장됨
        try:
            self.detection(source=image_path, file_id=file_id)
            self.logger.info("textDetection 함수 실행 성공")
        except Exception as e:
            self.logger.error(f"yolov5 detection 함수 실행 중 에러 발생: {e}")
            raise HTTPException(status_code=500, detail=f"yolov5 detection 함수 실행 중 에러 발생: {e}")

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

    async def save_temp_file(self, file, file_id) -> str:
        temp_file_path = f"temp_{file_id}.jpg"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        self.logger.info(f"임시 파일 저장 - 경로: {temp_file_path}")
        return temp_file_path

    async def send_cropped_images_to_ocr(self, dir_path: Path, remove_folder_path: Path, ocr_url: str) -> dict:
        
        categorized_data = {}
        
        # 크롭된 이미지들의 루트 폴더에서 하위 객체 폴더들 속 파일들에 대해 OCR 서버로 요청을 보내고 결과값을 반환
        for category_dir in dir_path.iterdir():
            if category_dir.is_dir():  # 하위 디렉토리만 처리
                self.logger.info(f"처리 중인 카테고리 디렉토리: {category_dir}")

                files_data = []
                open_files = []

                # 하위 디렉토리 내 이미지 파일들을 처리
                for file_path in category_dir.glob("*.jpg"):
                    f = open(file_path, "rb")
                    files_data.append(('files', (file_path.name, f, 'image/jpeg')))
                    open_files.append(f)

                # 크롭된 이미지 파일이 없는 경우
                if not files_data:
                    # 파일 객체 닫기
                    for f in open_files:
                        f.close()
                    self.logger.info(f"카테고리 '{category_dir}' 에 이미지 파일이 없습니다.")
                    continue

                # OCR 서버로 전송
                self.logger.info(f"{category_dir} 디렉토리의 이미지를 OCR 서버로 전송")
                async with httpx.AsyncClient() as client:
                    try:
                        timeout_limit = 45
                        response = await client.post(url=ocr_url, files=files_data, timeout=timeout_limit)
                        response.raise_for_status()
                        result_texts = response.json()

                        # 파일 이름 기준으로 정렬
                        sorted_keys = sorted(result_texts.keys())
                        sorted_data = {key: result_texts[key] for key in sorted_keys}
                        self.logger.info(f"{category_dir} 디렉토리의 OCR 결과: {sorted_data}")

                        # 보낸 파일과 받은 파일 비교
                        sent_file_names = [file_data[1][0] for file_data in files_data]
                        received_file_names = list(result_texts.keys())
                        missing_files = set(sent_file_names) - set(received_file_names)

                        if missing_files:
                            self.logger.warning(f"카테고리 '{category_dir}' 에서 누락된 파일: {missing_files}")
                        else:
                            self.logger.info(f"카테고리 '{category_dir}' 의 모든 파일이 성공적으로 처리되었습니다.")
                        
                        # 결과값 저장
                        categorized_data[category_dir.name] = sorted_data
                        self.logger.info(f"카테고리 '{category_dir}' 의 OCR 결과값 추가 성공")

                    finally:
                        # 파일 객체 닫기
                        for f in open_files:
                            f.close()
                        # 폴더 삭제
                        if os.path.exists(category_dir):
                            shutil.rmtree(category_dir)
                            self.logger.info(f"폴더 '{category_dir}' 가 성공적으로 삭제되었습니다.")
        
        # file_id에 대한 디렉토리 삭제
        if os.path.exists(remove_folder_path):
            shutil.rmtree(remove_folder_path)
            self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
        
        return categorized_data
        
        # self.logger.info(f"OCR 서버로 크롭된 이미지 전송 - 디렉토리 경로: {dir_path}, 제거할 폴더 경로: {remove_folder_path}, OCR URL: {ocr_url}")        
        # files_data = []
        # open_files = []
        # for file_path in dir_path.glob("*.jpg"):
        #     f = open(file_path, "rb")
        #     files_data.append(('files', (file_path.name, f, 'image/jpeg')))
        #     open_files.append(f)
        
        # # 송신한 정보들 로깅
        # self.logger.info(f"httpx를 통해 송신한 정보들: {files_data}")

        # # 크롭된 이미지 파일이 없는 경우
        # if not files_data:
        #     # 파일 객체 닫기
        #     for f in open_files:
        #         f.close()
        #     self.logger.info("모든 파일 객체를 닫았습니다.")
            
        #     # 폴더 삭제
        #     if os.path.exists(remove_folder_path):
        #         shutil.rmtree(remove_folder_path)
        #         self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
        #     else:
        #         self.logger.info(f"폴더 '{remove_folder_path}' 가 존재하지 않습니다.")
        #     self.logger.error("크롭된 이미지 파일이 존재하지 않습니다.")
            
        #     # 404 에러 발생
        #     raise HTTPException(status_code=404, detail="크롭된 이미지 파일이 존재하지 않습니다.")

        # self.logger.info(f"httpx 작업 전 - 수신지 url: {ocr_url}")
        # async with httpx.AsyncClient() as client:
        #     try:
        #         timeout_limit = 45
        #         response = await client.post(url=ocr_url, files=files_data, timeout=timeout_limit)
        #         response.raise_for_status()
        #         result_texts = response.json()
        #         self.logger.info("httpx를 통해 ocr 서버의 api로부터 리턴값 받음")
                
        #         # OCR 결과값들이 3, 4, 2, 5, 1... 등의 순서로 오는 경우가 있어서 정렬
        #         sorted_keys = sorted(result_texts.keys())
        #         sorted_data = {key: result_texts[key] for key in sorted_keys}
        #         self.logger.info(f"정렬된 json 데이터를 반환: {sorted_data}")
                
        #         # 원래 보낸 파일들의 이름을 추출
        #         sent_file_names = [file_data[1][0] for file_data in files_data]
        #         self.logger.info(f"송신한 파일 리스트: {sent_file_names}")

        #         # OCR 서버에서 처리된 파일들의 이름을 추출
        #         received_file_names = list(result_texts.keys())
        #         self.logger.info(f"OCR 서버에서 받은 파일 리스트: {received_file_names}")

        #         # 누락된 파일 확인
        #         missing_files = set(sent_file_names) - set(received_file_names)
        #         if missing_files:
        #             self.logger.warning(f"누락된 파일: {missing_files}")
        #         else:
        #             self.logger.info("모든 파일이 성공적으로 처리되었습니다.")
                
        #         return sorted_data

        #     except httpx.TimeoutException:
        #         self.logger.error("Request timeout while requesting server2")
        #         raise HTTPException(status_code=504, detail=f"Server2 did not respond in time. timeout limit is {timeout_limit} seconds.")
        #     except httpx.RequestError as exc:
        #         self.logger.error(f"Request error while requesting server2: {exc}")
        #         raise HTTPException(status_code=500, detail=f"Error while requesting server2: {exc}")
        #     except httpx.HTTPStatusError as exc:
        #         self.logger.error(f"HTTP error response from server2: {exc.response.text}")
        #         raise HTTPException(status_code=exc.response.status_code, detail=f"Error response from server2: {exc.response.text}")
        #     except Exception as exc:
        #         self.logger.error(f"Unexpected error: {exc}")
        #         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {exc}")
        #     finally:
        #         for f in open_files:
        #             f.close()
        #         self.logger.info("모든 파일 객체를 닫았습니다.")
        #         if os.path.exists(remove_folder_path):
        #             shutil.rmtree(remove_folder_path)
        #             self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
        #         else:
        #             self.logger.info(f"폴더 '{remove_folder_path}' 가 존재하지 않습니다.")
                