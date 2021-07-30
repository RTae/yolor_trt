from src.services.model_detection import Detection

class model:
    def __init__(self):
        self.detection = Detection()
    
    def inference(self, byte_image):
        log, result = self.detection.inference(byte_image)

        return None, result
