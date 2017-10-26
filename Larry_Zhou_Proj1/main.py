# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform

# name of the input file
imname = 'turkmen.tif'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]
b = b[0.05*height:0.95*height][0.05*height:0.95*height]
g = g[0.05*height:0.95*height][0.05*height:0.95*height]
r = r[0.05*height:0.95*height][0.05*height:0.95*height]
def ssd(img1, img2, x, y):
    rolled_img2 = np.roll(img2, x, 0)
    rolled_img2 = np.roll(rolled_img2, y, 1)
    return sum(sum((img1-rolled_img2)**2))
def ncc(img1, img2, x, y):
    rolled_img2 = np.roll(img2, x, 0)
    rolled_img2 = np.roll(rolled_img2, y, 1)
    flat_img1 = img1.flatten()
    flat_img2 = img2.flatten()
    norm1 = np.linalg.norm(flat_img1)
    norm2 = np.linalg.norm(flat_img2)
    norm_img1 = np.divide(flat_img1, norm1)
    norm_img2 = np.divide(flat_img2, norm2)
    return np.dot(norm_img1, norm_img2)
### ag = align(g, b)
### ar = align(r, b)
# for x in range(-15, 15):
#     for y in range(-15, 15):
# np.sum(g, b)
# np.sum(r, b)
check = sk.transform.rescale(b, 1)
def scale5(img1, img2, x1, x2, y1, y2, scale):
    level5_img1 = sk.transform.rescale(img1, scale)
    level5_img2 = sk.transform.rescale(img2, scale)
    min_ssd = 9999999999999
    min_x = 0
    min_y = 0
    for x in range(x1, x2):
        for y in range(y1, y2):
            curr_ssd = ssd(level5_img1, level5_img2, x, y)
            if curr_ssd < min_ssd:
                min_x = x
                min_y = y
                min_ssd = curr_ssd
    if scale == 1:
        return (min_x, min_y)
    return scale5(img1, img2, min_x*2-3, min_x*2+3, min_y*2-3, min_y*2+3, scale * 2)
resc = 1
## use pyramid until the length of the image is less than 200
while (len(check) > 200):
    resc = resc / 2
    check = sk.transform.rescale(b, resc)
# put initial parameters into recursive function
rx5, ry5 = scale5(b, r, -15, 15, -15, 15, resc)
gx5, gy5 = scale5(b, g, -15, 15, -15, 15, resc)

## did this part manually at first but did not work universally for both jpg and tif
# rx1, ry1 = scale5(b, r, -15, 15, -15, 15)
# rx2, ry2 = scale4(b, r, rx1*2 - 5, rx1*2 + 5, ry1*2 - 5, ry1*2 + 5)
# rx3, ry3 = scale3(b, r, rx2*2 - 4, rx2*2 + 5, ry2*2 - 5, ry2*2 + 5)
# rx4, ry4 = scale2(b, r, rx3*2 - 4, rx3*2 + 5, ry3*2 - 5, ry3*2 + 5)
# rx5, ry5 = scale1(b, r, rx4*2 - 4, rx4*2 + 5, ry4*2 - 5, ry4*2 + 5)
# gx1, gy1 = scale5(b, g, -15, 15, -15, 15)
# gx2, gy2 = scale4(b, g, gx1*2 - 5, gx1*2 + 5, gy1*2 - 5, gy1*2 + 5)
# gx3, gy3 = scale3(b, g, gx2*2 - 4, gx2*2 + 4, gy2*2 - 4, gy2*2 + 4)
# gx4, gy4 = scale2(b, g, gx3*2 - 4, gx3*2 + 4, gy3*2 - 4, gy3*2 + 4)
# gx5, gy5 = scale1(b, g, gx4*2 - 5, gx4*2 + 5, gy4*2 - 5, gy4*2 + 5)

print(rx5)
print(ry5)
print(gx5)
print(gy5)
ar = np.roll(r, rx5, 0)
ar = np.roll(ar, ry5, 1)
ag = np.roll(g, gx5, 0)
ag = np.roll(ag, gy5, 1)
# create a color image
im_out = np.dstack([ar, ag, b])
skio.imshow(im_out)
skio.show()
# save the image
fname = 'C:\\Users\\Larry\\Desktop\\cs194\\out_path\\turkmen_color.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()