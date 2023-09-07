#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:05:47 2023

@author: betulerkantarci
"""

import matplotlib.pyplot as plt
from skimage import io,restoration
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np


img = io.imread("camSensor.jpeg", as_gray=True)


from skimage.transform import rescale, resize, downscale_local_mean
rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)
resized_img = resize (img, (200,200))
downscaled_img = downscale_local_mean(img, (4,3))


from skimage.filters import roberts, sobel, scharr, prewitt
edge_roberts = roberts (downscaled_img)
#edge_sobel = sobel(downscaled_img)
#edge_scharr = scharr (downscaled_img)
#edge_prewitt = prewitt(downscaled_img)
plt.imshow(edge_roberts)

"""
from skimage.feature import canny
edge_canny = canny(downscaled_img , sigma=3)
plt.imshow(edge_canny)

"""

#img = io.imread("/Users/betulerkantarci/Desktop/nonr.png")
entropy_img = entropy(edge_roberts, disk(3))
plt.imshow(entropy_img, cmap='gray')

from skimage.filters import try_all_threshold

from skimage.filters import threshold_otsu
thresh = threshold_otsu(entropy_img)

binary = entropy_img >= thresh
plt.imshow(binary, cmap='gray')
print("The percent bright pixsels is: ", (np.sum(binary==1)*100)/(np.sum(binary==1)+np.sum(binary==0)))

