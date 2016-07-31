import os
import sys

import json
import boto3
import urllib
import StringIO
import hashlib

import logging
import tornado
from tornado.web import url

sys.path.append("{}/caffe/python".format(os.path.expanduser('~')))
import caffe

import numpy as np
import PIL.Image
import skimage.color as color
import scipy.ndimage.interpolation as sni

caffe.set_mode_gpu()
caffe.set_device(0)

############

# download model from https://www.dropbox.com/s/8iq5wm4ton5gwe1/colorization_release_v0.caffemodel
# and/or see demo notebook: https://github.com/richzhang/colorization
net = caffe.Net(
    '/home/ubuntu/model/colorization/colorization_deploy_v0.prototxt',
    '/home/ubuntu/model/colorization/colorization_release_v0.caffemodel',
    caffe.TEST
)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature

############

with open(os.path.expanduser('~') + '/keys.json') as f:
    s = f.read()
    keys = json.loads(s)

############

class GetImage(tornado.web.RequestHandler):
    # SUPPORTED_METHODS = ('GET')

    def get(self):
        input_image_url = self.get_argument('url')

        # generate a filename with the MD5 of input URL
        generated_filename = hashlib.md5(input_image_url).hexdigest() + '.jpg'

        s3 = boto3.Session(
            aws_access_key_id=keys['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=keys['AWS_SECRET_ACCESS_KEY'],
        ).resource('s3')

        ####
        # check if the file already exists; if so, exit early
        # http://stackoverflow.com/a/33843019
        exists = False

        try:
            s3.Object('colourful-past', generated_filename).load()
        except botocore.exceptions.ClientError as e:
            # if e.response['Error']['Code'] == "404":
                exists = False
        else:
            exists = True

        if exists:
            self.set_status(200)
            self.write("https://s3-us-west-2.amazonaws.com/colourful-past/{}".format(generated_filename))
            return

        ####
        
        input_filename = '/tmp/{}'.format(generated_filename)
        urllib.urlretrieve(input_image_url, input_filename)

        ####

        # this Caffe API wants a filename string and I'm too lazy to wrange skimage directly
        img_rgb = caffe.io.load_image(input_filename)
        img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
        img_l = img_lab[:,:,0] # pull out L channel
        (H_orig,W_orig) = img_rgb.shape[:2] # original image size

        # resize image to network input size
        img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
        img_lab_rs = color.rgb2lab(img_rs)
        img_l_rs = img_lab_rs[:,:,0]

        ####

        net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
        net.forward() # run network

        ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
        ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
        img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

        ####

        f = StringIO.StringIO()
        image_array = np.uint8(img_rgb_out*255)
        PIL.Image.fromarray(image_array).save(f, 'jpeg')

        ####
        
        s3.Bucket('colourful-past').put_object(
            Key=generated_filename,
            Body=f.getvalue(),
            ContentType='image/jpeg'
        )

        self.set_status(200)
        self.write("https://s3-us-west-2.amazonaws.com/colourful-past/{}".format(generated_filename))

############

tornado_settings = { 'debug': True, 'autoreload': True, 'port': 8000 }

def make_app(settings):
    return tornado.web.Application([
        url(r"/colour", GetImage),
    ], **tornado_settings)

def main():
    app = make_app(tornado_settings)
    app.listen(tornado_settings['port'])
    tornado.ioloop.IOLoop.current().start()

    logging.info("Tornado server listening on port {}".format(tornado_settings['port']))

if __name__ == '__main__':
    main()
