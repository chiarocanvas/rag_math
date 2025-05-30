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
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
            self.processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
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
                model="typhoon2-qwen2vl-7b-vision-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": r"""Ты — специализированный OCR для математических заданий. Твоя задача:
1. Точно и полностью выписать всё математическое задание с изображения, включая формулы, условия, текстовые пояснения, ограничения и т.д.
2. Не сокращай и не перефразируй текст — выпиши всё слово в слово, как на изображении.
3. Сохрани все математические символы, индексы, степени, знаки, пробелы и форматирование.
4. Не добавляй и не убирай ничего от себя.
5. Не пытайся решить, упростить или объяснить задание.

ПРАВИЛА ВЫВОДА:
- Верни только полный текст задания, без пояснений и комментариев.
- НЕ используй LaTeX-разделители ($, \\, \(, \), \[, \]) - выводи формулы как есть.
- НЕ добавляй обратные слеши перед математическими функциями (sin, cos, tan и т.д.).
- Сохрани все пробелы, переносы строк и форматирование как на изображении.
- Не добавляй и не убирай ни одного символа.
- Не исправляй опечатки, если они есть на изображении."""
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
            # Удаляем LaTeX-разделители
            result = result.replace('\\[', '').replace('\\]', '')
            result = result.replace('\\(', '').replace('\\)', '')
            result = result.replace('$$', '').replace('$', '')
            
            # Удаляем обратные слеши перед математическими функциями
            math_functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'log', 'ln', 'exp']
            for func in math_functions:
                result = result.replace(f'\\{func}', func)
            
            return result
