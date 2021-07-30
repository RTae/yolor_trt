import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import logging
from src.services.model.core.lp_net import get_pose_net, SoftArgmax2D


class PoseEsitmation():
    def __init__(self, 
                 model_weights = './src/utils/asserts/lp_net_50_256x192_with_gcb.pth.tar',
                 img_size = (192,256), 
                 threshold = 0.4,
                 ):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.img_size = img_size
        self.threshold = threshold
        logging.info(f'Model inference on {self.device}')

        self.model = get_pose_net()
        self.model.load_state_dict(torch.load(model_weights))
        self.model.eval()
        #self.model.to(self.device)
        self.beta_soft_argmax = SoftArgmax2D(beta=160)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def __box2cs(self, box, image_width, image_height):
        x, y, w, h = box[:4]
        return self.__xywh2cs(x, y, w, h, image_width, image_height)

    def __xywh2cs(self, x, y, w, h, image_width, image_height):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = 200

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    
    def __get_affine_transform( self, 
                                center,
                                scale,
                                rot,
                                output_size,
                                shift=np.array([0, 0], dtype=np.float32),
                                inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.__get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.__get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.__get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def __get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def __get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)


    def __get_max_preds(self, batch_heatmaps):
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]

        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

        maxvals = np.amax(heatmaps_reshaped, 2)
        maxvals = maxvals.reshape((batch_size, num_joints, 1))

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds = self.beta_soft_argmax(torch.from_numpy(batch_heatmaps)).numpy()
        preds *= pred_mask

        return preds, maxvals
    
    def __transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = self.__get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.__affine_transform(coords[p, 0:2], trans)
        return target_coords
    
    def __affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


    def __get_final_preds(self, batch_heatmaps, center, scale):
        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        preds, maxval = self.__get_max_preds(batch_heatmaps)

        # Transform back
        for i in range(preds.shape[0]):
            preds[i] = self.__transform_preds(preds[i], center[i], scale[i], [heatmap_width, heatmap_height])

        return preds[0], maxval[0]

    def preProcessing(self, img, bbox):
        c, s = self.__box2cs(bbox, self.img_size[0], self.img_size[1])
        r = 0
        trans = self.__get_affine_transform(c, s, r, self.img_size)
        input = cv2.warpAffine( img, trans, (self.img_size[0], self.img_size[1]), flags=cv2.INTER_LINEAR)
        input = self.transform(input).unsqueeze(0)
        
        return [input, c, s]
    
    def detect(self, data, c, s):
        with torch.no_grad():
            pred = self.model(data)
            preds, conf = self.__get_final_preds(pred.numpy(), np.asarray([c]), np.asarray([s]))
            result = np.concatenate([preds,conf], axis=1)

        return result

    def postProcessing(self, pred, img):
        for x, y, conf in pred:
            if conf > self.threshold:
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
        
        return img
    
    def inference(self, img, bbox):
        result = self.preProcessing(img, bbox)
        result = self.detect(*result)
        img = self.postProcessing(result, img)

        return img