from fastapi import APIRouter, File, UploadFile, HTTPException
import requests
import os, sys
from io import BytesIO
from pathlib import Path
import logging
import json
from PIL import Image

from app.service.yoloService import YOLOv5Service

from app.config import apikey, serverURL

# 로그 설정
# logging.config.fileConfig('app/config/logging_config.ini')
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
SERVER2_HEALTH_URL = serverURL.SERVER2_HEALTH_URL
SERVER2_OCR_MULTI_URL = serverURL.SERVER2_OCR_MULTI_URL    # 이민섭 ocr서버 api 주소

CLOVA_OCR_URL = serverURL.CLOVA_OCR_URL
CLOVA_SECRET_KEY = apikey.CLOVA_OCR_API_KEY

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


# 클로바 OCR 서버로 이미지를 받아서 작업하는 API
@yoloRouter.post("/yolo_clova", response_model=dict)
async def use_clovaOCR(file: UploadFile = File(...)):
    
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
    return await yolov5_service.send_cropped_images_to_clovaOCR(dir_path=dir_path, remove_folder_path=remove_folder_path, ocr_url=CLOVA_OCR_URL,  api_secret_key=CLOVA_SECRET_KEY)
    # return {"message": "success"} 



# 클로바 OCR을 한 번만 쓰는 API
@yoloRouter.post("/yolo_clova_once", response_model=dict)
async def use_clovaOCR(file: UploadFile = File(...)):
    
    # 임시 저장할 파일 경로
    file_id = temp_id.get_id()
    temp_file_path = await yolov5_service.save_temp_file(file, file_id)
    logger.info(f"임시 파일 경로: {temp_file_path}")
    
    # yolo로 이미지 크롭 수행\
    try:
        yolov5_service.textDetection(image_path=temp_file_path, file_id=file_id, save_csv=True, save_txt=True, save_crop=False, conf_thres=0.3)
    
    except Exception as e:
        logger.error(f"yolo로 이미지 크롭 수행 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="yolo로 이미지 크롭 수행 중 오류 발생")

        
    # # 클로바 OCR 서버로 전송
    try:
        clova_result = await yolov5_service.send_original_image_to_clovaOCR(img_path=temp_file_path, ocr_url=CLOVA_OCR_URL,  api_secret_key=CLOVA_SECRET_KEY)
        logger.info(f"클로바 OCR 서버로 이미지 전송 성공")
    except Exception as e:
        logger.error(f"클로바 OCR 서버로 이미지 전송 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="클로바 OCR 서버로 이미지 전송 중 오류 발생")
    
    # 전체적인 내용을 original_content에 저장
    original_content = ""
    if clova_result and 'images' in clova_result:
        for image_data in clova_result['images']:
            for field in image_data.get('fields', []):
                original_content += field.get('inferText', '') + " "
    
    # 확인용 파일 저장
    write_path = Path("yolov5") / "runs" / "detect" / file_id / "clova_result" / f"{file_id}.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)  # 상위 디렉토리까지 생성

    with open(write_path, 'w', encoding='utf-8') as f:
        json.dump(clova_result, f, ensure_ascii=False, indent=4)  # JSON으로 변환 후 저장
    logger.info(f"클로바 OCR 결과 저장 완료: {write_path}")
    
    # 이미지 파일의 크기 대입
    with Image.open(temp_file_path) as img:
        image_width, image_height = img.size

    # YOLO 텍스트 파일을 읽어 상대 좌표를 절대 좌표로 변환
    def parse_yolo_txt(yolo_txt_path):
        yolo_boxes = []
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                class_id = int(data[0])
                center_x = float(data[1]) * image_width
                center_y = float(data[2]) * image_height
                width = float(data[3]) * image_width
                height = float(data[4]) * image_height

                # YOLO의 (center_x, center_y, width, height)를 (x_min, y_min, x_max, y_max)로 변환
                x_min = center_x - width / 2
                y_min = center_y - height / 2
                x_max = center_x + width / 2
                y_max = center_y + height / 2
                yolo_boxes.append({"class_id": class_id, "bbox": (x_min, y_min, x_max, y_max)})
        return yolo_boxes

    # 클로바 OCR JSON 파일을 읽어와 텍스트 영역을 반환
    def parse_clova_json(clova_json_path):
        with open(clova_json_path, 'r', encoding='utf-8') as f:
            clova_data = json.load(f)
        clova_boxes = []
        for image in clova_data.get("images", []):
            for field in image.get("fields", []):
                vertices = field["boundingPoly"]["vertices"]
                x_min = vertices[0]["x"]
                y_min = vertices[0]["y"]
                x_max = vertices[2]["x"]
                y_max = vertices[2]["y"]
                infer_text = field["inferText"]
                clova_boxes.append({"bbox": (x_min, y_min, x_max, y_max), "text": infer_text})
        return clova_boxes

    # 두 박스의 IoU를 계산해 겹침 비율 반환
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 교집합 좌표 계산
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        # 교집합 넓이 계산
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # 각 박스 넓이 계산
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        # IoU 계산
        return inter_area / box2_area if box2_area else 0

    # YOLO 박스와 클로바 텍스트 영역을 비교해 N% 이상 겹칠 경우 포함
    def match_yolo_with_clova(yolo_boxes, clova_boxes, overlap_threshold=0.5):
        result = {}
        for yolo_box in yolo_boxes:
            class_id = yolo_box["class_id"]
            yolo_bbox = yolo_box["bbox"]
            included_texts = []

            for clova_box in clova_boxes:
                clova_bbox = clova_box["bbox"]
                infer_text = clova_box["text"]

                # 겹치는 비율 계산
                overlap_ratio = calculate_iou(yolo_bbox, clova_bbox)

                # 겹치는 비율이 설정된 threshold 이상인 경우 텍스트 포함
                if overlap_ratio >= overlap_threshold:
                    included_texts.append(infer_text)

            # 텍스트들을 문장 형태로 결합하여 저장
            if included_texts:
                class_name = yolov5_service.class_names.get(class_id, f"unknown_class_{class_id}")
                if class_name not in result:
                    result[class_name] = {}

                # 텍스트를 문장 형태로 결합하여 저장
                combined_text = ' '.join(included_texts)
                # "texts_1", "texts_2" 형태로 저장
                text_entry_key = f"texts_{len(result[class_name]) + 1}"
                result[class_name][text_entry_key] = combined_text

        return result

    
    # 파일 경로 설정
    yolo_txt_path = Path("yolov5") / "runs" / "detect" / file_id / "labels" / f"{file_id}.txt"
    clova_json_path = Path("yolov5") / "runs" / "detect" / file_id / "clova_result" / f"{file_id}.json"
    N = 0.5
    
    # YOLO, 클로바 OCR 데이터 파싱 및 매칭 수행
    yolo_boxes = parse_yolo_txt(yolo_txt_path)
    clova_boxes = parse_clova_json(clova_json_path)
    matched_data = match_yolo_with_clova(yolo_boxes, clova_boxes, overlap_threshold=N)
    
    result_data = {"original_content": original_content}
    result_data.update(matched_data)
    logger.info(f"탐지된 문장: {matched_data}")
    
    return result_data
