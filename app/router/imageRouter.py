## 이미지 URL을 입력으로 받아 Service로 넘겨주는 라우터
from fastapi import APIRouter
from pydantic import BaseModel
import os
import logging

# 로그 설정 파일을 로드하여 설정 적용
# logging.config.fileConfig('app/config/logging_config.ini')

# 로거 생성
logger = logging.getLogger(__name__)

imageRouter = APIRouter()

class Item(BaseModel):
    img_url: str
    
@imageRouter.post("/url")
def create_item(item: Item):
    # 이미지 URL을 받아와서 처리하는 로직
    # 여기에 실제로 이미지를 다운로드하거나 OCR을 수행하는 로직을 추가할 수 있습니다.
    logger.info(f"Received image URL: {item.img_url}")
    return {"img_url": item.img_url}
