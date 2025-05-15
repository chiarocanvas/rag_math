from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import Image
import logging
MAX_WIDTH = 1980
MAX_HEIGHT = 1080
class MathOCR:
    def __init__(self):
        """
        Инициализация модели и процессора.
        """
        try:
            logging.info("Загрузка модели и процессора для OCR...")
            self.model = load_model()
            self.processor = load_processor()
            logging.info("Модель и процессор успешно загружены.")
        except Exception as e:
            logging.error(f"Ошибка при инициализации MathOCR: {e}")
            raise RuntimeError("Не удалось загрузить модель OCR.")

    def infer_image(self, pil_image, temperature, type_ocr = 'texify') :
            pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)
            input_img = pil_image
            if  type_ocr == 'texify':
                model_output = batch_inference([input_img], self.model, self.processor, temperature=temperature)
                return model_output[0]
            elif type_ocr == 'vllm':
                pass
