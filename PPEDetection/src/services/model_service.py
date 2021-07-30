from src.services.model_detection import Detection

class model:
    def __init__(self):
        self.detection = Detection()
    
    def inference(self, byte_image):
        # Person detection
        log, result = self.detection.inference(byte_image)
        img = result[0]
        bbox = result[1]
        print(bbox)

        return None, img
