import matplotlib.pyplot as plt
from align_image_code import align_images
from scipy import misc
from scipy import ndimage
import numpy as np 
from sklearn.preprocessing import normalize
import skimage as sk
from scipy import sparse
from scipy import ndimage
import skimage.io as skio
import skimage.transform as skt
from skimage import color
from PIL import Image
from align_image_code import match_img_size
##hopefully I'm not missing any imports

# First load images

# high sf
im1 = plt.imread('./larry.jpg')/255.

# low sf
im2 = plt.imread('./wolf.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies


def lowPass(img1, sigma):
    return ndimage.filters.gaussian_filter(img1, sigma)


def highPass(img1, sigma):
    lowPassed = ndimage.filters.gaussian_filter(img1, sigma)
    return (img1 - lowPassed)

def hybrid_image(im1, im2, sigma1, sigma2):
   lowPassed = lowPass(im2, sigma2)
   highPassed = highPass(im1, sigma1)
   return highPassed + lowPassed

sigma1 = 10
sigma2 = 1

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
plt.imshow(hybrid)
plt.show()
misc.imsave("hybrid_test.png", hybrid)

## Part 1.3
## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
def pyramids(im, n):
    sigma = 1
    count = n
    def helper(img, sigma, n):
        if n > 0:
            a = ndimage.filters.gaussian_filter(img, sigma)
            b = img - a
            misc.imsave("gauss" + str(n) + ".jpg", a)
            misc.imsave("laplac" + str(n) + ".jpg", b)
            return helper(a, sigma*2, n-1)
    return helper(im, sigma, count)
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)