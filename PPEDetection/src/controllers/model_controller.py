from src.queue.tasks import inference
from src.services.model_service import helper

def controller_inference(image_byte):
    image_str = helper().imgByte2imgStr(image_byte)
    result = inference.delay(image_str)
    return result
