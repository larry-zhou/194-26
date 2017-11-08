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

# Part 1.1
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

#Part 1.2
im1 = plt.imread('./larry.jpg')/255.

# low sf
im2 = plt.imread('./wolf.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

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

##Part 1.4
def rgb2gray(rgb): # taken from online to convert color to gray
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def pyramids1(im, n):
    sigma = 1
    count = n
    def helper(img, sigma, n):
        if n > 0:
            a = ndimage.filters.gaussian_filter(img, sigma)
            b = img - a
            if n == 1:
                return b

            return helper(a, sigma*2, n-1)
    return helper(im, sigma, count)
def pyramids2(im, n):
    sigma = 1
    count = n
    def helper(img, sigma, n):
        if n > 0:
            a = ndimage.filters.gaussian_filter(img, sigma)
            if n == 1:
                return a
            # misc.imsave("gauss" + str(n) + ".png", a)
            # misc.imsave("lapalc" + str(n) + ".png", b)
            return helper(a, sigma*2, n-1)
    return helper(im, sigma, count)

img1 = plt.imread('./sun.jpg')/255
img2 = plt.imread('./moon.jpg')/255
img1, img2 = align_images(img1, img2)
height= img1.shape[0]
width= img1.shape[1]
L = np.ones((height, width/2 + 1)) #need this +1 for odd case
R = np.zeros((height, width/2 + 1))
a = np.concatenate((L, R), axis=1)
img1 = color.rgb2gray(img1)
img2 = color.rgb2gray(img2)
LA5 = pyramids1(img1, 5)
LB5 = pyramids1(img2, 5)
GR5 = pyramids2(a, 5)
LA4 = pyramids1(img1, 4)
LB4 = pyramids1(img2, 4)
GR4 = pyramids2(a, 4)
LA3 = pyramids1(img1, 3)
LB3 = pyramids1(img2, 3)
GR3 = pyramids2(a, 3)
LA2 = pyramids1(img1, 2)
LB2 = pyramids1(img2, 2)
GR2 = pyramids2(a, 2)
LA1 = pyramids1(img1, 1)
LB1 = pyramids1(img2, 1)
GR1 = pyramids2(a, 1)

def combine(la, lb, gr):
    height, width = la.shape
    ret = np.zeros((height, width))
    for i in range(0, len(la)):
        for j in range(0, len(la[0])):
            ret[i][j] = (gr[i][j] * la[i][j]) + ((1-gr[i][j]) * lb[i][j])
    return ret

LS5 = combine(LA5, LB5, GR5)
LS4 = combine(LA4, LB4, GR4)
LS3 = combine(LA3, LB3, GR3)
LS2 = combine(LA2, LB2, GR2)
LS1 = combine(LA1, LB1, GR1)

LS = LS5 + LS4 + LS3 + LS2 + LS1
fname = './blend.png'
plt.imshow(color.rgb2gray(LS), cmap="gray")
plt.show()