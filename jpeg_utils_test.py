import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s.%(msecs)04d [%(levelname).1s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

import simplejpeg as sjpeg

from jpeg_utils import *
    
def pe(a, b):
    return (abs(a - b) / a * 100)

with open('bird.jpg', 'rb') as f:
    jpeg = f.read()

img = sjpeg.decode_jpeg(jpeg)
for qf in range(50, 96):
    jpeg = sjpeg.encode_jpeg(img, qf, 'RGB', '444', False)
    blocks = jpeg_to_coef(jpeg)
    code_length = get_image_code_length(blocks)
    file_size = len(jpeg)
    logging.info(('QF: {} file_size: {}, code_length: {}, diff: {}, pe: {}'
                    .format(qf, file_size, code_length,
                    code_length - file_size, pe(file_size, code_length))))
