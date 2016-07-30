import os
import sys
import argparse

sys.path.append("{}/caffe/python".format(os.path.expanduser('~')))
import caffe

import numpy as np
import PIL.Image
import skimage.color as color
import scipy.ndimage.interpolation as sni

# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (12, 6)

caffe.set_mode_gpu()
caffe.set_device(0)

############

# download model from https://www.dropbox.com/s/8iq5wm4ton5gwe1/colorization_release_v0.caffemodel
# and/or see demo notebook: https://github.com/richzhang/colorization
net = caffe.Net('/home/ubuntu/model/colorization/colorization_deploy_v0.prototxt', '/home/ubuntu/model/colorization/colorization_release_v0.caffemodel', caffe.TEST)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature

############

parser = argparse.ArgumentParser(description='colourise black and white image')
parser.add_argument('inputfile', metavar='I', type=str,
                    help='input image file')
parser.add_argument('outputfile', metavar='O', type=str,
                    help='filename to write output')
args = parser.parse_args()

input_file = os.path.join(os.getcwd(), args.inputfile)

############

img_rgb = caffe.io.load_image(input_file)
img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# create grayscale version of image (just for displaying)
# img_lab_bw = img_lab.copy()
# img_lab_bw[:,:,1:] = 0
# img_rgb_bw = color.lab2rgb(img_lab_bw)

# resize image to network input size
img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]

############

net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
net.forward() # run network

ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

############

output_file = os.path.join(os.getcwd(), args.outputfile)

#image_array = np.uint8(np.clip(img_rgb_out, 0, 255))
image_array = np.uint8(img_rgb_out*255)

f = open(output_file, 'w')
PIL.Image.fromarray(image_array).save(f, 'jpeg')

