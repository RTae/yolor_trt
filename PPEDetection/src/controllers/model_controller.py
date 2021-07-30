from src.queue.tasks import inference
from src.utils.helper.model_utils import imgByte2imgStr

def controller_inference(image_byte):
    image_str = imgByte2imgStr(image_byte)
    result = inference.delay(image_str)
    return result
