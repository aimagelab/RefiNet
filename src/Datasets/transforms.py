import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs


class GaussianNoise(object):
    def __init__(self, image_size, sigma=1, check: bool = True):
        self.check = check
        self.image_size = image_size
        self.sigma = sigma

    def __call__(self, kpt):
        done = False
        while done is False:
            noise = np.random.normal(0, self.sigma, (kpt.shape[0], kpt.shape[1]))
            kpt = kpt + noise
            if self.check:
                if kpt[kpt[:, 0] < 0].sum() > 0 or kpt[kpt[:, 1] < 0].sum() > 0:
                    done = False
                elif kpt[kpt[:, 0] > self.image_size[1]].sum() > 0 or kpt[kpt[:, 1] > self.image_size[0]].sum() > 0:
                    done = False
                else:
                    done = True
            done = True
        return kpt
