import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import normalize
import skimage as sk
from scipy import sparse
from align_image_code import align_images
from scipy import ndimage
import skimage.io as skio
import skimage.transform as skt
from skimage import color
from PIL import Image

target = plt.imread('./orange.jpeg')/255 
source = plt.imread('./apple.jpeg')/255 
mask = plt.imread('./orange_mask.jpeg')/255 

#split the images into their respective channels
target_blue = target[:,:,0]
target_green = target[:,:,1]
target_red = target[:,:,2]
source_blue = source[:,:,0]
source_green = source[:,:,1]
source_red = source[:,:,2]
height = source.shape[0]
width = source.shape[1]
mn = height * width
im2var = np.zeros(mn)
mask = sk.color.rgb2gray(mask)

for i in range(0, len(im2var)):
    im2var[i] = i 
im2var = im2var.reshape((height, width))
A1 = sparse.identity(mn, format='lil')
b_vect1 = np.zeros(mn)
b_vect1 = b_vect1.reshape((mn), 1)
A2 = sparse.identity(mn, format='lil')
b_vect2 = np.zeros(mn)
b_vect2 = b_vect2.reshape((mn), 1)
A3 = sparse.identity(mn, format='lil')
b_vect3 = np.zeros(mn)
b_vect3 = b_vect3.reshape((mn), 1)

for y in range(0, height-1):
    for x in range(0, width-1):
        if mask[y][x] < 1:
            b_vect1[im2var[y][x]] = target_blue[y][x]
            b_vect2[im2var[y][x]] = target_green[y][x]
            b_vect3[im2var[y][x]] = target_red[y][x]
        else:
            A1[im2var[y][x], im2var[y][x]] = 4
            A1[im2var[y][x], im2var[y+1][x]] = -1
            A1[im2var[y][x], im2var[y-1][x]] = -1
            A1[im2var[y][x], im2var[y][x] + 1] = -1
            A1[im2var[y][x], im2var[y][x] - 1] = -1
            b_vect1[im2var[y][x]] = 4 * source_blue[y][x] - source_blue[y+1][x] - source_blue[y][x+1] - source_blue[y-1][x] - source_blue[y][x-1]
            A2[im2var[y][x], im2var[y][x]] = 4
            A2[im2var[y][x], im2var[y+1][x]] = -1
            A2[im2var[y][x], im2var[y-1][x]] = -1
            A2[im2var[y][x], im2var[y][x] + 1] = -1
            A2[im2var[y][x], im2var[y][x] - 1] = -1
            b_vect2[im2var[y][x]] = 4 * source_green[y][x] - source_green[y+1][x] - source_green[y][x+1] - source_green[y-1][x] - source_green[y][x-1]
            A3[im2var[y][x], im2var[y][x]] = 4
            A3[im2var[y][x], im2var[y+1][x]] = -1
            A3[im2var[y][x], im2var[y-1][x]] = -1
            A3[im2var[y][x], im2var[y][x] + 1] = -1
            A3[im2var[y][x], im2var[y][x] - 1] = -1
            b_vect3[im2var[y][x]] = 4 * source_red[y][x] - source_red[y+1][x] - source_red[y][x+1] - source_red[y-1][x] - source_red[y][x-1]
csr1 = sparse.csr_matrix(A1)
v1 = sparse.linalg.spsolve(csr1, b_vect1)
v1 = v1.reshape(height, width)
csr2 = sparse.csr_matrix(A2)
v2 = sparse.linalg.spsolve(csr2, b_vect2)
v2 = v2.reshape(height, width)
csr3 = sparse.csr_matrix(A3)
v3 = sparse.linalg.spsolve(csr3, b_vect3)
v3 = v3.reshape(height, width)
target[:,:,0] = v1
target[:,:,1] = v2
target[:,:,2] = v3
target = target/target.max()
plt.imshow(target)
plt.show()
# skio.imshow(v)
# skio.show()
fname = './apple_orange.jpg'
skio.imsave(fname, target)