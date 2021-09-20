import numpy as np
import cv2
import scipy.signal as sig
def espec_warp(img,tForm):
    # calc transformed image
    imgArea0 = tForm['imgArea0']
    H = tForm['H']
    newImgSize = tForm['newImgSize']
    imgArea1 = tForm['imgArea1']
    img = img - np.median(img)
    img = sig.medfilt2d(img,kernel_size=(3,3))
    imgCountsPerArea = img/imgArea0
    imgCountsPerArea[imgArea0==0] =0
    imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
    imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

    im_out = cv2.warpPerspective(imgCountsPerArea, H, newImgSize)*imgArea1
    return im_out