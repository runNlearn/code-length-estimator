import math
import functools

import numpy as np

import jpegio

from turbojpeg import TurboJPEG
from turbojpeg import TJPF_RGB, TJSAMP_420, TJSAMP_444, TJFLAG_ACCURATEDCT
tjpeg = TurboJPEG()

from huffman import *
from jpeg2dct.numpy import loads


_zigzag_index = (
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)

TO_ZIGZAG_INDEX = np.argsort(_zigzag_index)


def encode_jpeg(img, qf, subsampling=False):
   subsample = TJSAMP_420 if subsampling else TJSAMP_444
   jpeg = tjpeg.encode(img, quality=qf, pixel_format=TJPF_RGB,
                       jpeg_subsample=subsample, flags=TJFLAG_ACCURATEDCT)
   return jpeg


def decode_jpeg(jpeg):
    img = tjpeg.decode(jpeg, pixel_format=TJPF_RGB,
                       flags=TJFLAG_ACCURATEDCT)
    return img


def jpeg_to_coef(jpeg):
    coefs = loads(jpeg, False)  # `False`: get quantized dct coefficient
    return np.stack(coefs, axis=0)


def np_delta_encode(coefs):
    coefs[..., 1:, 0] = coefs[..., 1:, 0] - coefs[..., :-1, 0]


def np_raster_scan(coefs):
    return coefs[..., TO_ZIGZAG_INDEX]


def run_length_encode(ac_coef):
    non_zero_indices = np.where(ac_coef != 0)[0]
    num_zeros = non_zero_indices.copy()
    num_zeros[1:] = non_zero_indices[1:] - non_zero_indices[:-1] - 1
    result = []
    for num, idx in zip(num_zeros, non_zero_indices):
        while num >= 16:
            result.append((15, 0))
            num -= 16
        result.append((num, ac_coef[idx]))
    result.append((0, 0))
    return result


def get_dc_bit_length(dc_value, encode_huff_tbl):
    required_length = 0 if dc_value == 0 else int(math.log2(abs(dc_value))) + 1
    return required_length + len(encode_huff_tbl[required_length])


def get_ac_bit_length(ac_symbol, encode_huff_tbl):
    num_zeros, value = ac_symbol
    required_length = 0 if value == 0 else int(math.log2(abs(value))) + 1
    return (len(encode_huff_tbl[16 * num_zeros + required_length])
            + required_length)


def get_block_bit_length(block, dc_huff_tbl, ac_huff_tbl):
    bit_length = get_dc_bit_length(block[0], dc_huff_tbl)
    ac_symbols = run_length_encode(block[1:])
    bit_length += sum(get_ac_bit_length(symbol, ac_huff_tbl) for symbol
                       in ac_symbols) 
    return bit_length


def get_image_code_length(coefs):
    func_y = functools.partial(get_block_bit_length,
                               dc_huff_tbl=Y_DC_HUFF_TBL,
                               ac_huff_tbl=Y_AC_HUFF_TBL)
    func_c = functools.partial(get_block_bit_length,
                               dc_huff_tbl=C_DC_HUFF_TBL,
                               ac_huff_tbl=C_AC_HUFF_TBL)
        
    coefs = coefs.reshape([3, -1, 64])
    np_delta_encode(coefs)
    coefs = np_raster_scan(coefs)

    y = sum(map(func_y, coefs[0]))
    cb = sum(map(func_c, coefs[1]))
    cr = sum(map(func_c, coefs[2]))

    return math.ceil((y + cb + cr) / 8)

def tile(coef):
  h, w = coef.shape[:2]
  coef = np.reshape(coef, (h, w, 8, 8))
  coef = np.transpose(coef, (0, 2, 1, 3))
  coef = np.reshape(coef, (h * 8, w * 8))
  return coef

def encode_jpeg_from_qdct(coefs):
  """ Generate JPEG encoded file from the quantized dct coefficients
    Args:
      coefs: a list of dct coefficients, (y, cb, cr).
        dtype of coefficients have to be `np.int32`.
    Returns:
      JPEG bytes string 
  """
  y_coef  = tile(coefs[0])
  cb_coef = tile(coefs[1])
  cr_coef = tile(coefs[2])
  coef_arrays = [y_coef, cb_coef, cr_coef]

  if y_coef.shape == cb_coef.shape:
    subsampling = False
  else: # 420
    subsampling = True

  dummy1_path = '.dummy1.jpg'
  dummy2_path = '.dummy2.jpg'
  size = y_coef.shape[0]
  dummy = np.zeros((size, size, 3), dtype=np.uint8)
  with open(dummy1_path, 'wb') as f:
    f.write(encode_jpeg(dummy, 1, subsampling=subsampling))
  jobj = jpegio.read(dummy1_path)
  for i in range(2):
    np.copyto(jobj.quant_tables[i], np.ones((8, 8), dtype=np.int32))
  for i in range(3):
    np.copyto(jobj.coef_arrays[i], coef_arrays[i])
  jobj.write(dummy2_path)
  with open(dummy2_path, 'rb') as f:
    jpeg = f.read()
  return jpeg
