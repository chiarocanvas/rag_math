# from texify.inference import batch_inference
# from texify.model.model import load_model
# from texify.model.processor import load_processor
from PIL import Image
import logging
from openai import OpenAI
import io
import base64
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
MAX_WIDTH = 1980
MAX_HEIGHT = 1080
class MathOCR:
    def __init__(self):
        """
        Инициализация модели и процессора.
        """
        try:
            logging.info("Загрузка модели и процессора для OCR...")

            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            logging.info("Модель и процессор успешно загружены.")
        except Exception as e:
            logging.error(f"Ошибка при инициализации MathOCR: {e}")
            raise RuntimeError("Не удалось загрузить модель OCR.")
   

    def infer_image(self, pil_image, type_ocr='vlm'):
        pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
        input_img = pil_image
        img_byte_arr = io.BytesIO()
        input_img.save(img_byte_arr, format="JPEG")
        image_bytes = img_byte_arr.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        if type_ocr == 'vlm':
            completion = self.client.chat.completions.create(
                model="nanonets.nanonets-ocr-s",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": r"""Ты — специализированный OCR для математических заданий. Твоя задача:
1. Точно и полностью выписать всё математическое задание с изображения, включая формулы, условия, текстовые пояснения, ограничения и т.д.
2. Не добавляй и не убирай ничего от себя.
3. Верни только  уравнение, без пояснений и комментариев.
."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
            )
            result = completion.choices[0].message.content
            print(result)
            return result
