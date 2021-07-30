import io
import cv2
import torch
import base64
import numpy as np
from imageio import imread
from src.services.trt_loader import TrtModel
from src.utils.helper.model_utils import letterbox, non_max_suppression, drawBBox

class Detection:
    def __init__(self, 
                model_weights = './src/utils/asserts/yolor_csp_x_star-fp16.trt', 
                imgsz = 896, 
                threshold = 0.4,
                iou_thres = 0.6,
                names = './src/utils/asserts/coco.names'):
        '''
        Model config
        model_weights : weight of model ,default /src/assert/yolor_csp_x_star.qunt.onnx
        max_size : max size of image (widht, height) ,default 896
        names : name of class ref ,default coco/src/assert/coco.names
        '''

        self.colors = (255,0,0)
        self.imgsz = (imgsz, imgsz)
        self.threshold = threshold
        self.iou_thres = iou_thres

        # load model
        self.model = TrtModel(model_weights, imgsz)

    def preProcessing(self, img_string):

        bgr_img = imread(io.BytesIO(base64.b64decode(img_string)))

        ## Prepocessing image before feed to model
        # Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
        # BGR to RGB
        inp = inp[:, :, ::-1].transpose(2, 0, 1)
        # Normalization from 0 - 255 (8bit) to 0.0 - 1.0
        inp = inp.astype('float32') / 255.0
        # Expand dimention to have batch size 1
        inp = np.expand_dims(inp, 0)

        return None, [bgr_img, inp]
    
    def detect(self, image):
        '''
        Object detection from coco label
        model name: YOLOR_CSP_X
        image : input image that already prepocessing 
        '''
        pred = self.model.run(image)[0]
        return None, pred


    def postProcessing(self, image_d, image_p, pred):
        '''
        After get result from model this function will post processing the result before
        seat a output
        '''

        # NMS
        with torch.no_grad():
            pred = non_max_suppression(torch.tensor(pred), conf_thres=self.threshold, iou_thres=self.iou_thres)
        
        # Filter class only person
        det = pred[0]
        det = det[det[:,5] == 0,:]

        # Check have prediction
        if det is not None and len(det):
            # Rescale boxes from img_size to origin size
            _, _, height, width = image_p.shape
            h, w, _ = image_d.shape
            det[:, 0] *= w/width
            det[:, 1] *= h/height
            det[:, 2] *= w/width
            det[:, 3] *= h/height
            for x1, y1, x2, y2, _ , _ in det:
                # Draw BBox
                image_d = drawBBox((x1, y1), (x2, y2), image_d, self.colors)

        # Convert to string image
        _, im_buf_arr = cv2.imencode(".jpg", image_d)
        imgd = base64.b64encode(im_buf_arr).decode()

        return None, imgd

    def inference(self, img_string):
        log, image_list = self.preProcessing(img_string)
        log, pred = self.detect(image_list[1])
        log, result = self.postProcessing(image_list[0], image_list[1], pred)

        return None, result