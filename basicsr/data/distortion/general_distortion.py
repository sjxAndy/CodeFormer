import random
import os.path as osp
import os
import cv2
import glob
from .base import DistortionBase
from . import augmentations as A
from .utils import distortion_utils as D



class GeneralDistortion(DistortionBase):
    """
    GFRNet模型通用退化类，适用于参考图模型和无参考图模型
    """
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.base_dir = osp.dirname(osp.abspath(__file__))
        self.kernel_files = glob.glob(self.base_dir + "/kernels/*.mat")
        # 记录各种退化函数的力度
        self.distortion_max_score = 16.
    

    def __call__(self, img):
        self.distortion_record.clear()
        src_img = img.copy()
        h, w, _ = src_img.shape
        use_sharpen = True
        chance = random.random()

        # motion blur: 1/3的概率
        if chance < 1./3:
            if random.random() > 0.7:
                img = D.motionBlur(img, self.kernel_files, record=self.distortion_record)
            else:
                img = D.motionBlurSimple(img, self.config.get('motionBlurSimple').get("motion", [2, 9]), 
                                         record=self.distortion_record)
        # motion blur + defocus: 1/6概率
        elif chance >= 1./3 and chance < 1./2:
            if random.random() > 0.5:
                img = D.motionBlurSimple(img, motion=self.config.get('motionBlurSimple').get("motion", [2, 9]), 
                                         record=self.distortion_record)
            else:
                img = img = D.motionBlur(img, self.kernel_files, record=self.distortion_record)
            img = D.defocusBlur(img, defocus=self.config.get('defocusBlur').get("defocus", [3, 9]), record=self.distortion_record)
        elif chance < 2./3:
            use_sharpen = False
        else:
            img = D.defocusBlur(img, defocus=self.config.get('defocusBlur').get("defocus", [3, 9]), record=self.distortion_record)

        img = D.downsample(img, scale=self.config.get('downsample').get("scale", [1, 4]), record=self.distortion_record)

        # 50%概率加高斯噪声，50%概率加raw噪声
        if random.random() >= 0.5:
            # 在Y通道加高斯噪声
            if random.random() > 0.75:
                img = D.ychannelNoise(img, sigma=self.config.get('ychannelNoise').get("sigma", [0, 8]),
                                      denoise_prob=self.config.get('ychannelNoise').get("denoise_prob", 0.5))
            # 直接加高斯噪声
            else:
                img = D.gaussianNoise(img, var=self.config.get('gaussianNoise').get("var", [0, 8]),
                                      denoise_prob=self.config.get('gaussianNoise').get("denoise_prob", 0.5),
                                      record=self.distortion_record)
        else:
            img = D.rawNoiseAndSharpen(img, iso_ranges=self.config.get('rawNoiseAndSharpen').get("iso_ranges", [800, 12000]),
                                      use_sharp=use_sharpen)
        
        # JPEG压缩
        img = D.jpegCompression(img, jpegq=self.config.get('jpegCompression').get("jpegq", [30, 95]),
                                prob=self.config.get('jpegCompression').get("prob", 0.35),
                                record=self.distortion_record)
        
        # 美颜
        if hasattr(self.config, "beautify"):
            img = D.beautify(img, strength=self.config.get('beautify').get("strength", [0.08, 1.2]), prob=self.config.get('beautify').get('prob', 0.0))

        # 上采样回原始分辨率
        if img.shape != src_img.shape:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # CutBlur数据增强
        if hasattr(self.config, "cutblur"):
            img = A.cutblur(img, src_img, ratio=self.config.get('cutblur').get("ratio", 0.3),
                           prob=self.config.get('cutblur').get("prob", 0.2))


        # 部分退化分数需要归一化
        for k in self.distortion_record.keys():
            if k in ["motionBlurSimple", "defocusBlur"]:
                self.distortion_record[k] /= self.distortion_max_score
        
        return img





