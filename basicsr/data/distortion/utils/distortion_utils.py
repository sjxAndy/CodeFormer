#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import sys


from skimage.draw import disk
from scipy.ndimage import rotate
from ..augmentations.img_utils import advancedSharpen
from .noise_sim import add_raw_noise
from .HybridDenoise.hybrid_denoise import hybrid_denoise

np.seterr(divide='ignore', invalid='ignore')

import random
import cv2


########################
# helper func
########################
def gamma(img, gamma=1.1):
    """
    对图片进行gamma变换
    args:
        img: [HWC], 数值在0~255, uint8
    """
    img = img.astype(np.float32) / 255.
    img = img ** gamma
    img = img.clip(0, 1)
    img = img * 255.
    return img.astype(np.uint8)


def adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


def diskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord + 1

    rr, cc = disk(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3 or dim == 5):
        kernel = adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel


def nonLocalMean(img):
    """
    Y通道进行nonlocal mean降噪处理
    args:
        img: [HWC], 数值在0~255, uint8
    """
    h = random.randint(5, 10)
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    y = img_ycbcr[:, :, 0]
    y = cv2.fastNlMeansDenoising(y, h=h, templateWindowSize=7, searchWindowSize=21)
    img_ycbcr[:, :, 0] = y

    img = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2RGB)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def lap_pyr(img, num_level=5):

    # Gaussian Pyramid
    layer = img.copy()
    gaussian_pyramid = [layer]
    for i in range(num_level + 1):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)

    # Laplacian Pyramid
    layer = gaussian_pyramid[num_level]
    laplacian_pyramid = [layer]
    for i in range(num_level, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid


def lap_pyr_rec(laplacian_pyramid, var=5, num_level=5):
    reconstructed_image = laplacian_pyramid[0]

    _var_list = [var + x for x in range(num_level)]
    for i in range(1, num_level + 1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
        h, w, c = reconstructed_image.shape
        gaussian_noise = np.random.normal(0, _var_list[i - 1], (h, w, c))
        reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])

        reconstructed_image = np.clip(reconstructed_image + gaussian_noise, 0.0, 255.0).astype(np.uint8)
    return reconstructed_image


########################
# distort ops
########################


def defocusBlur(img, type='uniform', defocus=5, radius=15, ratio=1, prob=1.0, record=None):
    """
    对图片做失焦模糊
    args:
        img: [HWC], 数值在0~255, uint8
        type: 失焦类型, 支持uniform或者circle
        defocus: 失焦kernel, 在失焦类型为circle的时候生效, 
                数据类型为list or int, list的情况下会随机从列表中挑选一个力度
        radius: 失焦半径, 在失焦类型为uniform的时候生效
        ratio: 失焦kernel放大系数, 大图建议用更大的系数
        prob: 概率值,  0~1
        record: 外部传入的字典, 用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        if type == 'uniform':
            x, y = np.mgrid[-(radius + 1):(radius + 1), -(radius + 1):(radius + 1)]
            disk = (np.sqrt(x ** 2 + y ** 2) < radius).astype(np.float)
            disk /= disk.sum()
            if record is not None:
                _function_name = sys._getframe().f_code.co_name
                record[_function_name] = 2 * radius + 1
        elif type == 'circle':
            if isinstance(defocus, list):
                assert len(defocus) == 2
                left = defocus[0]
                right = int(np.ceil(defocus[1] * ratio))
                defocusKernelDims = [i for i in range(left, right, 2)]
                kerneldim = random.sample(defocusKernelDims, k=1)[0]
                disk = diskKernel(dim=kerneldim)
            else:
                disk = diskKernel(dim=defocus)
        else:
            raise NotImplementedError("type %s is not implemented in defocusBlur" % type)
        
        img = cv2.filter2D(img, -1, disk)
        
    return img


def downsample(img, scale, prob=1.0, record=None):
    """
    对图片做随机下采样
    args:
        img: [HWC], 数值在0~255, uint8
        scale: list or int, list的情况下会随机从列表中挑选一个力度
        prob: 概率值， 0~1
        record: 外部传入的字典，用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        if isinstance(scale, list):
            assert len(scale) == 2
            if isinstance(scale[1], int):
                scale = random.randint(scale[0], scale[1])
            else:
                scale = random.uniform(scale[0], scale[1])

        if record is not None:
            _function_name = sys._getframe().f_code.co_name
            record[_function_name] = scale ** 2

        if scale <= 1:
            return img

        new_h = img.shape[0] // scale
        new_w = img.shape[1] // scale
        return cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    else:
        return img


def upsample(img, scale, prob=1.0):
    """
    对图片做随机上采样
    args:
        img: [HWC], 数值在0~255, uint8
        scale: list or int, list的情况下会随机从列表中挑选一个力度
        prob: 概率值， 0~1
    """
    if random.random() < prob:
        if isinstance(scale, list):
            scale = random.randint(scale[0], scale[1])
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        return img


def gaussianBlur(img, strength, bias=0, prob=1.0):
    """
    对图片进行高斯模糊
    args:
        img: [HWC], 数值在0~255, uint8
        strength: list or int, 模糊力度, list的情况下会随机从列表中挑选一个力度
        prob: 概率值， 0~1
    """
    if random.random() < prob:
        if isinstance(strength, list):
            left = strength[0]
            right = strength[1]
            sigma = random.randint(left, right) + bias
        else:
            strength = int(strength)
        sizeG = sigma * 12 + 1

        if sigma >= 1 and sigma <= 6:
            sizeG = sigma * 12 + 1
        return cv2.GaussianBlur(img, (sizeG, sizeG), sigma)
    else:
        return img


def gaussianNoise(img, var, denoise_prob=1.0, prob=1.0, record=None):
    """
    对图片加高斯噪声，并以一定概率开启降噪
    args:
        img: [HWC], 数值在0~255, uint8
        var: list or int, 方差, list的情况下会随机从列表中挑选一个方差
        prob: 概率值， 0~1
        record: 外部传入的字典，用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        if isinstance(var, list):
            left, right = var[0], var[1]
            var = random.randint(left, right)
        
            if record is not None:
                _function_name = sys._getframe().f_code.co_name
                record[_function_name] = float(var - left) / float(right - left)

        if var == 0:
            return img

        # 拉普拉斯图上加高斯噪声会带来异常退化，遂去除
            # # 50%概率在拉普拉斯图上加高斯噪声
            # if random.random() <= 0.5:
            #     laplacian_pyramid = lap_pyr(img)
            #     noise_img = lap_pyr_rec(laplacian_pyramid, var=var)
            # else:
        h, w, c = img.shape
        mean = 0

        gauss = np.random.normal(mean, var, (h, w, c))
        gauss = gauss.reshape(h, w, c)
        noise_img = gauss + img
        noise_img = noise_img.clip(0, 255).astype(np.uint8)

        if random.random() < denoise_prob:
            noise_img = nonLocalMean(img)
        return noise_img
    else:
        return img



def jpegCompression(img, jpegq, prob=1.0, record=None):
    """
    对图片做jpeg压缩
    args:
        img: [HWC], 数值在0~255，uint8
        jpegq: list or int，压缩力度，list的情况下会随机从列表中挑选一个方差
        prob: 概率值， 0~1
        record: 外部传入的字典，用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        if isinstance(jpegq, list):
            jpegq = random.randint(jpegq[0], jpegq[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpegq]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if record is not None:
            _function_name = sys._getframe().f_code.co_name
            record[_function_name] = 1.0 - (float(jpegq - 0) / 100.)
        return cv2.imdecode(encimg, 1)
    else:

        return img


def motionBlurSimple(img, motion, ratio=1., prob=1.0, record=None):
    """
    对图片做运动模糊, 只有直线运动轨迹
    args:
        img: [HWC], 数值在0~255, uint8
        motion: list or int, 压缩力度, list的情况下会随机从列表中挑选一个方差
        ratio: motion放大系数, 大图建议用更大的系数
        prob: 概率值,  0~1
        record: 外部传入的字典, 用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        if isinstance(motion, list):
            left = motion[0]
            right = int(np.ceil(motion[1] * ratio))
            degree = random.randint(left, right)
        else:
            degree = motion

        if record is not None:
            _function_name = sys._getframe().f_code.co_name
            record[_function_name] = degree
        
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel /= motion_blur_kernel.sum()

        img = gamma(img, 2.2)
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = gamma(img, 1 / 2.2)

    return img


def motionBlur(img, kernel_files, ratio=1., prob=1.0, record=None):
    """
    对图片做运动模糊，使用事先生成的模糊核，运动轨迹更复杂
    args:
        img: [HWC], 数值在0~255, uint8
        kernel_files: 模糊核文件
        ratio: motion放大系数, 大图建议用更大的系数
        prob: 概率值,  0~1
        record: 外部传入的字典, 用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        kernel_changes = list(np.linspace(1.0, 2.0, 5))
        kernel_angles = list(np.linspace(0, 360, 9))
        blur_kernel = scipy.io.loadmat(random.choice(kernel_files))['kernel']
        w, h = blur_kernel.shape
        ratio = random.choice(kernel_changes) * ratio
        new_shape = (int(w * ratio), int(h * ratio))
        blur_kernel = cv2.resize(blur_kernel, new_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=0)
        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=1)
        if random.random() > 0.5:
            blur_kernel = rotate(blur_kernel, random.choice(kernel_angles))
        
        blur_kernel /= blur_kernel.sum()
        img = gamma(img, 2.2)
        img = cv2.filter2D(img, -1, blur_kernel)
        img = gamma(img, 1 / 2.2)

        if record is not None:
            # max blur kernel: 35
            _function_name = sys._getframe().f_code.co_name
            record[_function_name] = new_shape[0] / (ratio * 2 * 35)

    return img


def beautify(img, strength, prob=1.0, record=None):
    """
    对图片进行美颜平滑处理，优先0.3的概率开启弱美颜，0.4的概率开启强美颜
    args:
        img: [HWC], 数值在0~255，uint8
        strength: 美颜力度, list or int
        prob: 概率值， 0~1
        record: 外部传入的字典，用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        pro = random.random()
        if pro > 0.7:
            if isinstance(strength, list):
                eps = (random.random() * (strength[1] - strength[0]) + strength[0]) * 0.1
            else:
                eps = strength
            
            r = random.randint(1, 4)
            img = img.astype(np.float32) / 255.
            img = (cv2.ximgproc.guidedFilter(img, img, r, eps ** 2, -1) * 255.0).astype(np.uint8)
        elif pro > 0.6:
            k, sigma = random.choice([5, 7, 9]), random.choice([35, 55, 65, 75, 85])
            img = cv2.bilateralFilter(img, k, sigma, sigma)
        else:
            pass

    return img


def ychannelNoise(img, sigma, ratio=2, denoise_prob=1.0, prob=1.0, record=None):
    """
    对图片Y通道加噪声, 并按照一定概率开启降噪
    args:
        img: [HWC], 数值在0~255, uint8
        sigma: 噪声方差, list or int
        ratio: 放大系数
        prob: 概率值， 0~1
        record: 外部传入的字典，用于记录当前退化函数的退化力度分数
    """
    if random.random() < prob:
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        row, col, _ = img_ycbcr.shape

        mean = 0
        if isinstance(sigma, list):
            sigma = random.randint(sigma[0], int(sigma[1] * ratio))
        else:
            sigma = sigma
        
        scale = random.randint(1, 3)

        if sigma == 0:
            return img
        
        gauss = np.random.normal(mean, sigma, (int(row / scale), int(col / scale), 1))
        gauss = gauss.reshape(int(row / scale), int(col / scale), 1)
        gauss = cv2.resize(gauss, (col, row), interpolation=cv2.INTER_LINEAR)

        noise_y = img_ycbcr[:, :, 0] + gauss
        noise_y = np.clip(noise_y, 0, 255)
        img_ycbcr[:, :, 0] = noise_y
        
        img = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCR_CB2RGB)
        img = np.clip(img, 0, 255).astype(np.uint8)

        if random.random() < denoise_prob:
            img = nonLocalMean(img)


    return img




def rawNoise(img, iso_ranges, noise_model="MI_NOTE10_NOISE", prob=1.0):
    """
    对图片加raw噪声
    args:
        img: [HWC], 数值在0~255，uint8
        iso_ranges: iso范围, list
        noise_model: 噪声模型，MI_NOTE10_NOISE | VIVO_NOISE
        prob: 概率值， 0~1
    """
    if random.random() < prob:
        img = add_raw_noise(img, iso_ranges, noise_model)
    
    return img


def rawNoiseAndSharpen(img, iso_ranges, noise_model="MI_NOTE10_NOISE", use_sharp=False, prob=1.0):
    """
    对图片加raw噪声，并按一定概率降噪和锐化
    args:
        img: [HWC], 数值在0~255，uint8
        iso_ranges: iso范围, list
        noise_model: 噪声模型，MI_NOTE10_NOISE | VIVO_NOISE
        use_sharp: 是否使用锐化
        prob: 概率值， 0~1
    """
    if random.random() < prob:
        img = add_raw_noise(img, iso_ranges, noise_model)

        # Y channel denoise
        if random.random() > 0.2:
            r = random.random()
            if 0.8 < r:
                image_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
                y_channel = image_ycbcr[:, :, 0]

                # Op non local mean parameters
                h_ = random.choice([5, 7, 9])
                sw_ = random.choice([19, 21, 23])
                tw_ = random.choice([5, 7, 9])

                y_channel = cv2.fastNlMeansDenoising(y_channel, h=h_, templateWindowSize=tw_, searchWindowSize=sw_)
                image_ycbcr[:, :, 0] = y_channel
                img = cv2.cvtColor(image_ycbcr, cv2.COLOR_YCR_CB2RGB)
                img = np.clip(img, 0, 255).astype(np.uint8)
            elif 0.6 < r and r <= 0.8:
                img = hybrid_denoise(img, nr_method="lap_pyr_rec", blend_alpha=0.8)
            elif 0.4 < r and r <= 0.6:
                img = hybrid_denoise(img, nr_method="median", blend_alpha=0.8)
            elif 0.2 < r and r <= 0.4:
                img = hybrid_denoise(img, nr_method="freq", blend_alpha=0.8)
            else:
                img = hybrid_denoise(img, nr_method="bilateral", blend_alpha=0.8)
            
            if random.random() > 0.5 and use_sharp:
                img = advancedSharpen(img)
    
    return img
