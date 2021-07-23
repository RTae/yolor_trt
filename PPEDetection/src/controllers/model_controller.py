from src.tasks.tasks import inference

def controller_inference(image_byte):
    result = inference(image_byte)
    return result
