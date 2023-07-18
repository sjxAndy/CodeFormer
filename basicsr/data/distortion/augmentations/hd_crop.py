import numpy as np
import random
import cv2
import math
from copy import deepcopy


def hd_crop(img, landmarks, target_size=None, prob=1.0):
    """
    根据关键点crop出更大的人脸，并保持原始分辨率
    img: np ndarray or list of np ndarray, 如果是list，则landmarks也必须是list
        [HWC]，数值在0~255，uint8
    landmarks: 285关键点，np.ndarray or list of np.ndarray
    target_size: crop后需要resize的分辨率
    prob: 概率值， 0~1
    """
    def _crop_face(_img, _landmark, target_size):
        src_h, src_w = _img.shape[:2]
        x_min = int(_landmark[:, 0].min())
        x_max = int(_landmark[:, 0].max())
        y_min = int(_landmark[:, 1].min())
        y_max = int(_landmark[:, 1].max())
        _img = _img[y_min:y_max, x_min:x_max]
        if target_size is None:
            target_size = (src_w, src_h)
        _img = cv2.resize(_img, target_size, interpolation=cv2.INTER_LINEAR)
        return _img

    if random.random() < prob:
        if isinstance(img, list):
            assert isinstance(landmarks, list)
            for i in range(len(img)):
                img[i] = _crop_face(img[i], landmarks[i], target_size)
        else:
            img = _crop_face(img, landmarks, target_size)

    return img



def hd_crop_facial_coordinates(landmark, coordinate, x_scale=1.0, y_scale=1.0):
    """
    根据hd_crop，对五官bbox进行坐标平移以及尺度缩放，landmark和coordinate必须是同个比例
    args:
        landmark: np.ndarray
        coordinates: np.ndarray(x1, y1, x2, y2)
        x_scale: 横坐标缩放比例
        y_scale: 纵坐标缩放比例
    """
    x_min = int(landmark[:, 0].min())
    y_min = int(landmark[:, 1].min())

    new_coord = deepcopy(coordinate)
    new_coord[0] = (new_coord[0] - x_min) * x_scale
    new_coord[1] = (new_coord[1] - y_min) * y_scale
    new_coord[2] = (new_coord[2] - x_min) * x_scale
    new_coord[3] = (new_coord[3] - y_min) * y_scale

    return new_coord


def hd_crop_landmark(landmark, x_scale=1.0, y_scale=1.0):
    """
    根据hd_crop，对关键点进行坐标平移
    args:
        landmark: np.ndarray
        x_scale: 横坐标缩放比例
        y_scale: 纵坐标缩放比例
    """
    new_landmark = deepcopy(landmark)
    x_min = int(landmark[:, 0].min())
    y_min = int(landmark[:, 1].min())
    new_landmark[:, 0] -= x_min
    new_landmark[:, 1] -= y_min
    new_landmark[:, 0] *= x_scale
    new_landmark[:, 1] *= y_scale

    return new_landmark

