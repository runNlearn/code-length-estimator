import math
import functools

import numpy as np
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

_to_zigzag_index = np.argsort(_zigzag_index)


def jpeg_to_coef(jpeg):
    coefs = loads(jpeg, False)  # `False`: get quantized dct coefficient
    return np.stack(coefs, axis=0)


def np_delta_encode(coefs):
    coefs[..., 1:, 0] = coefs[..., 1:, 0] - coefs[..., :-1, 0]


def np_raster_scan(coefs):
    return coefs[..., to_zigzag_index]


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


def get_dc_code_length(dc_value, encode_huff_tbl):
    required_length = 0 if dc_value == 0 else int(math.log2(abs(dc_value))) + 1
    return required_length + len(encode_huff_tbl[required_length])


def get_ac_code_length(ac_symbol, encode_huff_tbl):
    num_zeros, value = ac_symbol
    required_length = 0 if value == 0 else int(math.log2(abs(value))) + 1
    return len(encode_huff_tbl[16 * num_zeros + required_length]) + required_length


def get_block_code_length(block, dc_huff_tbl, ac_huff_tbl):
    code_length = get_dc_code_length(block[0], dc_huff_tbl)
    ac_symbols = run_length_encode(block[1:])
    code_length += sum(get_ac_code_length(symbol, ac_huff_tbl) for symbol
                       in ac_symbols)
    return code_length


def get_image_code_length(coefs):
    coefs = coefs.reshape([3, -1, 64])
    np_delta_encode(coefs)
    coefs = np_raster_scan(coefs)
    func = functools.partial(get_block_code_length,
                             dc_huff_tbl=Y_DC_HUFF_TBL,
                             ac_huff_tbl=Y_AC_HUFF_TBL)
    y = sum(map(func, coefs[0]))
    cb = sum(map(func, coefs[1]))
    cr = sum(map(func, coefs[2]))
    return y + cb + cr


def tf_delta_encode(coefs):
    ac = coefs[..., 1:]
    dc = coefs[..., 0:1]
    dc = tf.concat([dc[..., 0:1, :], dc[..., 1:, :] - dc[..., :-1, :]], axis=-2)
    return tf.concat([dc, ac], axis=-1)


def tf_raster_scan(coefs):
    return tf.gather(coefs, to_zigzag_index, axis=-1, batch_dims=0)


def tf_get_block_code_length(block):
    func = functools.partial(get_block_code_length,
                             dc_huff_tbl=Y_DC_HUFF_TBL,
                             ac_huff_tbl=Y_AC_HUFF_TBL)
    code_length = tf.numpy_function(func, [block], tf.int64)
    return code_length

