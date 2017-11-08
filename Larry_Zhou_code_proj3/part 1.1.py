import matplotlib.pyplot as plt
import numpy
from align_image_code import align_images
from scipy import misc
from scipy import ndimage

im1 = plt.imread('./David.jpg')/255.

def lowPass(img1, sigma):
    return ndimage.filters.gaussian_filter(img1, sigma)


def highPass(img1, sigma):
    lowPassed = ndimage.filters.gaussian_filter(img1, sigma)
    return (img1 - lowPassed)

def unblur(img):
    im = img
    filt = lowPass(im, 4)
    diff = im - filt
    diff *= .8
    ub = im + diff
    maximum = ub.max()
    ub = ub/maximum
    plt.imshow(ub)
    plt.show()
unblur(im1)
