import numpy as np
import cv2
import random
import colour_demosaicing


########################
# helper func
########################
def gamma(img, gamma=1.1):
    """
    对图片进行gamma变换
    args:
        img: [HWC], 数值在0~255，uint8
    """
    img = img.astype(np.float32) / 255.
    img = img ** gamma
    img = img.clip(0, 1)
    img = img * 255.
    return img.astype(np.uint8)



VIVO_NOISE = {
    '100iso': [[0.10302168, 0.22796172],
               [0.1037082, 0.20281659],
               [0.09871322, 0.34510456]],
    '500iso': [[0.51525423, 1.04507114],
               [0.51331967, 1.17793258],
               [0.48696414, 1.22589979]],
    '800iso': [[0.80884628, 2.43097323],
               [0.81739142, 2.38013651],
               [0.76519675, 2.49142857]],
    '1600iso': [[1.59314132, 7.88423861],
                [1.59319833, 8.21943797],
                [1.50654706, 8.25159436]],
    '3200iso': [[3.18628265, 7.88423861],
                [3.18639667, 8.21943797],
                [3.01309412, 8.25159436]]
}

MI_NOTE10_NOISE = {
    '100iso': [[0.0641, 0.2927], [0.0608, 0.2816], [0.0623, 0.3453]],
    '300iso': [[0.1927, 0.4707], [0.1759, 0.424], [0.1759, 0.424]],
    '800iso': [[0.5353, 1.1777], [0.4554, 1.2451], [0.5192, 0.9204]],
    '1600iso': [[1.0976, 3.1551], [0.863, 3.9203], [1.0158, 3.0116]],
    '3200iso': [[2.1691, 10.7588], [1.7578, 11.4467], [1.9651, 11.6351]],
    '4800iso': [[3.2640, 23.4308], [2.5920, 23.9907], [2.9760, 25.9715]],
    '12000iso': [[8.1600, 143.1812], [6.4800, 142.5315], [7.4500, 161.4467]],
    'r': {'s': 0.068, 'r0': 0.0099, 'r1': 0.6212},
    'b': {'s': 0.054, 'r0': 0.0098, 'r1': 1.4115},
    'g': {'s': 0.062, 'r0': 0.0112, 'r1': 0.1667},
}


def noise_meta_func(iso_, s, r0, r1):
    return iso_ / 100.0 * s, (iso_ / 100.0) ** 2 * r0 + r1


def add_raw_noise(img, iso_ranges, noise_model="MI_NOTE10_NOISE"):
    """
    对图片加raw noise
    A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising
    https://arxiv.org/abs/2003.12751
    
    args:
        img: [HWC], 数值在0~255，uint8
        iso_range: list, iso的范围
        noise_model: MI_NOTE10_NOISE or VIVO_NOISE
    """

    assert len(iso_ranges) == 2

    if noise_model == "MI_NOTE10_NOISE":
        n_p_collect = MI_NOTE10_NOISE
    elif noise_model == "VIVO_NOISE":
        n_p_collect = VIVO_NOISE
    else:
        raise NotImplementedError("noise model %s is not supported.")
    
    iso = random.randint(iso_ranges[0], iso_ranges[1])

    r_meta, b_meta, g_meta = \
        n_p_collect['r'], n_p_collect['b'], n_p_collect['g']
    n_p = []
    n_p.append(noise_meta_func(iso, r_meta['s'], r_meta['r0'], r_meta['r1']))
    n_p.append(noise_meta_func(iso, b_meta['s'], b_meta['r0'], b_meta['r1']))
    n_p.append(noise_meta_func(iso, g_meta['s'], g_meta['r0'], g_meta['r1']))

    raw = np.zeros(img.shape[:2], dtype=np.float32)

    # inverse gamma
    img = np.power(img.astype(np.float32) / 255, 2.2)

    # resampling
    raw[::2, ::2] = img[::2, ::2, 0] # R
    raw[1::2, ::2] = img[1::2, ::2, 1]  # G
    raw[::2, 1::2] = img[::2, 1::2, 1]  # G
    raw[1::2, 1::2] = img[1::2, 1::2, 2] # B

    # inverse WB
    awb_r = 0.5 * random.random() + 1.9
    awb_b = 0.4 * random.random() + 1.5
    raw[::2, ::2] /= awb_r
    raw[1::2, 1::2] /= awb_b

    # Assume now already subtracted black level
    raw = np.clip((raw * 1023), 0, 1023)

    # Possion, different possion for different color channel
    r = raw[::2, ::2]
    g1 = raw[1::2, ::2]  # two g is identical till this step
    g2 = raw[::2, 1::2]  # two g is identical till this step
    b = raw[1::2, 1::2]
    gamma_r, beta_r = n_p[0][0], n_p[0][1]
    gamma_g, beta_g = n_p[2][0], n_p[2][1]
    gamma_b, beta_b = n_p[1][0], n_p[1][1]

    noise_r = np.sqrt(gamma_r * r + beta_r) * np.random.normal(0, 1, r.shape)
    noise_g1 = np.sqrt(gamma_g * g1 + beta_g) * np.random.normal(0, 1, g1.shape)
    noise_g2 = np.sqrt(gamma_g * g2 + beta_g) * np.random.normal(0, 1, g2.shape)
    noise_b = np.sqrt(gamma_b * b + beta_b) * np.random.normal(0, 1, b.shape)

    raw[::2, ::2] += noise_r  # R
    raw[1::2, ::2] += noise_g1  # G
    raw[::2, 1::2] += noise_g2  # G
    raw[1::2, 1::2] += noise_b  # B

    # AWB
    raw[::2, ::2] *= awb_r  # awb_r
    raw[1::2, 1::2] *= awb_b  # awb_b
    raw = np.clip(raw, 0, 1023).astype(np.uint16)

    demosaicked_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(raw, 'RGGB')
    demosaicked_rgb = np.clip(demosaicked_rgb / 1023, 0, 1)

    img = np.power(demosaicked_rgb, 1 / 2.2)
    img = np.clip(img * 255., 0, 255).astype(np.uint8)

    return img
