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
#im2var[row][column]

#Part 2.1
source = plt.imread('./toy_problem.png')/255 
height = source.shape[0] 
width = source.shape[1]
mn = height * width
im2var = np.zeros(mn)
for i in range(0, len(im2var)):
    im2var[i] = i 
im2var = im2var.reshape((height, width))
A = np.zeros(((mn*2)+ 1) * mn)
A = A.reshape((((mn*2)+ 1), mn))
b_vect = np.zeros((mn*2)+1)
b_vect  = b_vect.reshape((mn*2)+1, 1)
pixel_index = 0
A[pixel_index][im2var[0][0]] = 1
b_vect[pixel_index] = source[0][0]
for y in range(0, height-1):
    for x in range(0, width-1):
        pixel_index +=1
        A[pixel_index][im2var[y][x+1]] = 1
        A[pixel_index][im2var[y][x]] = -1
        b_vect[pixel_index] = source[y][x+1] - source[y][x]
for y in range(0, height-1):
    for x in range(0, width-1):
        pixel_index +=1
        A[pixel_index][im2var[y+1][x]] = 1
        A[pixel_index][im2var[y][x]] = -1
        b_vect[pixel_index] = source[y+1][x] - source[y][x]
csr = sparse.csr_matrix(A)
v = sparse.linalg.lsqr(csr, b_vect)
v = np.array(v[0])
v = v.reshape(height, width)
plt.imshow(color.rgb2gray(v), cmap="gray")
plt.show()
# skio.imshow(v)
# skio.show()
fname = 'C:\\Users\\Larry\\Desktop\\toy.png'
skio.imsave(fname, v)
