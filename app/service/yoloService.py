import logging
import logging.config
import os
import shutil
from pathlib import Path
import httpx
from fastapi import HTTPException
from yolov5 import detection

import requests
import uuid
import time
import json


# 로그 설정
logging.config.fileConfig('app/config/logging_config.ini')

class YOLOv5Service:
    def __init__(self):
        self.detection = detection.run
        self.logger = logging.getLogger(__name__)
        self.logger.info("YOLOv5Service 인스턴스 생성됨")

    def textDetection(self, image_path, file_id, save_csv=False, save_txt=False, save_crop=True, conf_thres=0.6):
        self.logger.info(f"textDetection 함수 실행 - 이미지 경로: {image_path}, 파일 아이디: {file_id}")
        # 크롭된 이미지들이 경로에 저장됨
        try:
            # 추가할 매개변수가 뭐가 있을까...
            # save_txt, save_crop, conf_thres, name
            self.detection(source=image_path, file_id=file_id, save_csv=save_csv, save_txt=save_txt, save_crop=save_crop, conf_thres=conf_thres)
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
        temp_file_path = f"{file_id}.jpg"
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
                        # 파일 객체 닫기
                        for f in open_files:
                            f.close()
                        # # 카테고리 폴더 삭제
                        # if os.path.exists(category_dir):
                        #     shutil.rmtree(category_dir)
                        #     self.logger.info(f"폴더 '{category_dir}' 가 성공적으로 삭제되었습니다.")
        
        # file_id에 대한 디렉토리 삭제
        if os.path.exists(remove_folder_path):
            shutil.rmtree(remove_folder_path)
            self.logger.info(f"폴더 '{remove_folder_path}' 및 가 성공적으로 삭제되었습니다.")
        
        return categorized_data




    # 클로바 OCR API 사용
    async def send_cropped_images_to_clovaOCR(self, dir_path: Path, remove_folder_path: Path, ocr_url: str, api_secret_key: str) -> dict:
        self.logger.info(f"send_cropped_images_to_clovaOCR 함수 실행 - 이미지 경로: {dir_path}, 삭제할 폴더 경로: {remove_folder_path}, OCR 서버 URL: {ocr_url})")

        categorized_data = {}
        
        # 클로바 OCR 요청 URL 및 비밀 키
        api_url = ocr_url
        secret_key = api_secret_key
        
        # 크롭된 이미지 파일들을 클로바 OCR로 전송
        for category_dir in (d for d in dir_path.iterdir() if d.is_dir() and os.listdir(d)):
            self.logger.info(f"처리 중인 카테고리 디렉토리: {category_dir}")
            
            responses = []
            
            # 카테고리별 이미지 파일 처리
            for file_path in category_dir.glob("*.jpg"):
                image_file = file_path
                
                # 클로바 OCR 요청 JSON
                request_json = {
                    'images': [
                        {
                            'format': 'jpg',
                            'name': file_path.name,
                        }
                    ],
                    'requestId': str(uuid.uuid4()),
                    'version': 'V2',
                    'timestamp': int(round(time.time() * 1000))
                }

                # 요청 데이터 생성
                payload = {'message': json.dumps(request_json).encode('UTF-8')}
                files = [('file', open(image_file, 'rb'))]
                
                # 헤더에 비밀 키 추가
                headers = {
                    'X-OCR-SECRET': secret_key
                }

                # 요청 보내기
                try:
                    response = requests.post(api_url, headers=headers, data=payload, files=files)
                    response.raise_for_status()  # 요청이 실패하면 HTTPError 발생
                    responses.append(response.json())  # 응답을 JSON으로 변환하여 저장
                    self.logger.info(f"OCR 결과 응답: {response.json()}")
                except requests.exceptions.RequestException as exc:
                    self.logger.error(f"Request error while requesting Clova OCR API: {exc}")
                    raise HTTPException(status_code=500, detail=f"Error while requesting Clova OCR API: {exc}")
                finally:
                    # 파일 닫기
                    files[0][1].close()
            
            categorized_data[category_dir.name] = responses  # 카테고리별로 응답 저장
            self.logger.info(f"카테고리 '{category_dir}' 의 OCR 결과값 추가 성공")

        # 임시 파일 삭제
        if os.path.exists(remove_folder_path):
            shutil.rmtree(remove_folder_path)
            self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
        
        return categorized_data


    # 클로바 OCR API 한 번만 사용
    async def send_original_image_to_clovaOCR(self, img_path: Path, remove_folder_path: Path, ocr_url: str, api_secret_key: str) -> dict:
        self.logger.info(f"send_cropped_images_to_clovaOCR 함수 실행 - 이미지 경로: {dir_path}, 삭제할 폴더 경로: {remove_folder_path}, OCR 서버 URL: {ocr_url})")

        categorized_data = {}
        
        # 클로바 OCR 요청 URL 및 비밀 키
        api_url = ocr_url
        secret_key = api_secret_key
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 크롭된 이미지 파일들을 클로바 OCR로 전송
        for category_dir in (d for d in dir_path.iterdir() if d.is_dir() and os.listdir(d)):
            self.logger.info(f"처리 중인 카테고리 디렉토리: {category_dir}")
            
            responses = []
            
            # 카테고리별 이미지 파일 처리
            for file_path in category_dir.glob("*.jpg"):
                image_file = file_path
                
                # 클로바 OCR 요청 JSON
                request_json = {
                    'images': [
                        {
                            'format': 'jpg',
                            'name': file_path.name,
                        }
                    ],
                    'requestId': str(uuid.uuid4()),
                    'version': 'V2',
                    'timestamp': int(round(time.time() * 1000))
                }

                # 요청 데이터 생성
                payload = {'message': json.dumps(request_json).encode('UTF-8')}
                files = [('file', open(image_file, 'rb'))]
                
                # 헤더에 비밀 키 추가
                headers = {
                    'X-OCR-SECRET': secret_key
                }

                # 요청 보내기
                try:
                    response = requests.post(api_url, headers=headers, data=payload, files=files)
                    response.raise_for_status()  # 요청이 실패하면 HTTPError 발생
                    responses.append(response.json())  # 응답을 JSON으로 변환하여 저장
                    self.logger.info(f"OCR 결과 응답: {response.json()}")
                except requests.exceptions.RequestException as exc:
                    self.logger.error(f"Request error while requesting Clova OCR API: {exc}")
                    raise HTTPException(status_code=500, detail=f"Error while requesting Clova OCR API: {exc}")
                finally:
                    # 파일 닫기
                    files[0][1].close()
            
            categorized_data[category_dir.name] = responses  # 카테고리별로 응답 저장
            self.logger.info(f"카테고리 '{category_dir}' 의 OCR 결과값 추가 성공")

        # 임시 파일 삭제
        if os.path.exists(remove_folder_path):
            shutil.rmtree(remove_folder_path)
            self.logger.info(f"폴더 '{remove_folder_path}' 가 성공적으로 삭제되었습니다.")
        
        return categorized_data