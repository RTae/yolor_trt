from src.queue.tasks import inference
from src.services.model_service import model

def controller_inference(image_byte):
    image_str = model().imgByte2imgStr(image_byte)
    result = inference.delay(image_str)
    return result
