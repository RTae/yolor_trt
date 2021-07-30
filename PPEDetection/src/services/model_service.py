import io
import cv2
import logging
import base64
from imageio import imread
from src.services.model.model_detection import Detection
from src.services.model.model_pose_estimation import PoseEsitmation

class model:
    def __init__(self):
        #self.detection = Detection()
        self.pose = PoseEsitmation()

    def preProcessing(self, img_string):
        bgr_img = imread(io.BytesIO(base64.b64decode(img_string)))

        return None, bgr_img

    def detect(self, bgr_img):
        #log, result = self.detection.inference(bgr_img)
        #img = result[0]
        x = [[858, 91, 1614 - 858, 1060 - 91]]
        
        for box in x:
            bgr_img = self.pose.inference(bgr_img, box)

        return None, bgr_img

    def postProcessing(self, imgd):
        # Convert numpy image to string image
        _, im_buf_arr = cv2.imencode(".jpg", imgd)
        imgd = base64.b64encode(im_buf_arr).decode()

        return None, imgd
    
    def inference(self, byte_image):
        log, bgr_img = self.preProcessing(byte_image)
        log, img = self.detect(bgr_img)
        log, img = self.postProcessing(img)

        return None, img
