import tensorflow as tf
import numpy as np

import simplejpeg as sjpeg

from jpeg_utils import *
from huffman import *

IMAGE_SIZE = 224
CROP_PADDING = 32

def _decode_and_center_crop(image_bytes, image_size=IMAGE_SIZE):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3,
                                        dct_method='INTEGER_ACCURATE')
  image = tf.image.resize(image, [image_size, image_size], 'bicubic')
  return tf.cast(image, 'uint8')


def np_extract_coef_block(jpeg):
    coef = jpeg_to_coef(jpeg)
    coef_shape = coef.shape
    ch = np.random.randint(0, coef_shape[0], ())
    x = np.random.randint(0, coef_shape[2], ())
    y = np.random.randint(0, coef_shape[1], ())
    block = coef[ch, y, x]
    return block

def tf_process(jpeg):
    img = _decode_and_center_crop(jpeg)
    return img

def log_scale(value, round=True):
    value = np.log2(np.abs(value) + 1)
    value = np.round(value) if round else value
    return value

def np_process(img):
    qf = np.random.randint(50, 100)
    jpeg = sjpeg.encode_jpeg(img, qf, 'RGB', '444', False)
    block = np_extract_coef_block(jpeg)
    block = np_raster_scan(block)
    code_length = get_block_code_length(block, Y_DC_HUFF_TBL, Y_AC_HUFF_TBL)
    block = block.astype('float32')
    block = log_scale(block)
    code_length = code_length
    return block, code_length

def np_test_process(img):
    qf = np.random.randint(50, 100)
    jpeg = sjpeg.encode_jpeg(img, qf, 'RGB', '444', False)
    coef = jpeg_to_coef(jpeg)
    code_length = get_image_code_length(coef)
    blocks = coef.reshape([-1, 64])
    blocks = log_scale(blocks)
    code_length_with_header = len(jpeg) * 8
    return blocks, code_length, code_length_with_header
