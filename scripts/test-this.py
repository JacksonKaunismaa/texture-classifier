import cv2
from mahotas.features.lbp import lbp
import numpy as np


def local_binary(imname):
    an_im = cv2.imread(imname, cv2.IMREAD_COLOR)
    np_im = np.array(an_im)
    result1 = lbp(np_im.max(axis=2), 3, 10)
    result1 /= result1.sum()
    result2 = lbp(np_im.mean(axis=2), 2, 10)
    result2 /= result2.sum()
    result = np.concatenate((result1, result2))
    return result

ours = local_binary("./non_texture/100.jpg")
print(ours.shape)
