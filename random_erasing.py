from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
import numpy as np
import torch

def beta_mix_factor(num, factor):
    random_numbers = [np.random.beta(factor, factor) for _ in range(num)]
    total = sum(random_numbers)
    scaled_numbers = [num / total for num in random_numbers]
    return scaled_numbers
    
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class ModerateRLE(object):
    def __init__(self, probability = 0.5, beta_factor=0.5):
        self.probability = probability
        self.beta_factor = beta_factor
         
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        else:
            f = beta_mix_factor(3, self.beta_factor)
            tmp_img = f[0] * img[0,:,:] + f[1] * img[1,:,:] + f[2] * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img

class RadicalRLE(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, min_out=1e-6, beta_factor=0.5, sl=0.02, sh=0.4, r1=0.3, eps=1e-12):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.min_out = min_out
        self.eps = eps
        self.beta_factor = beta_factor

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        mask = torch.ones(img.shape)
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    alphar = np.random.beta(self.beta_factor, self.beta_factor)
                    alphag = np.random.beta(self.beta_factor, self.beta_factor)
                    alphab = np.random.beta(self.beta_factor, self.beta_factor)
                    maxr = (1 / (torch.max(img[0, x1:x1 + h, y1:y1 + w])))
                    maxg = (1 / (torch.max(img[1, x1:x1 + h, y1:y1 + w])))
                    maxb = (1 / (torch.max(img[2, x1:x1 + h, y1:y1 + w])))
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, x1:x1 + h, y1:y1 + w] * maxr * alphar
                    img[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w] * maxg * alphag
                    img[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w] * maxb * alphab
                    mask[0, x1:x1 + h, y1:y1 + w] = mask[0, x1:x1 + h, y1:y1 + w] * maxr * alphar
                    mask[1, x1:x1 + h, y1:y1 + w] = mask[1, x1:x1 + h, y1:y1 + w] * maxg * alphag
                    mask[2, x1:x1 + h, y1:y1 + w] = mask[2, x1:x1 + h, y1:y1 + w] * maxb * alphab
                else:
                    alpha = np.random.beta(0.5, 0.5)
                    maxr = 1 / torch.max(img[0, x1:x1 + h, y1:y1 + w])
                    img[0, x1:x1 + h, y1:y1 + w] = img[0, x1:x1 + h, y1:y1 + w] * maxr * alpha
                    mask[0, x1:x1 + h, y1:y1 + w] = mask[0, x1:x1 + h, y1:y1 + w] * maxr * alpha
                min_flag = torch.min(mask.view(-1), 0)[0]
                if min_flag < self.min_out:
                    return img
        return img