from PIL import Image

import jpegio


__all__ = [
    'Y_DC_HUFF_TBL',
    'Y_AC_HUFF_TBL',
    'C_DC_HUFF_TBL',
    'C_AC_HUFF_TBL',
]

dummy_img = Image.new('RGB', (32, 32), color='orange')
dummy_img.save('dummy.jpg', subsampling=False, quality=1)
dummy_jobj = jpegio.read('dummy.jpg')

y_dc, c_dc = dummy_jobj.dc_huff_tables
y_ac, c_ac = dummy_jobj.ac_huff_tables

def expand_huffman_tbl(counts, symbols, encoding=True, hex=False):
    huff_tbl = dict()
    bits = ['']
    idx = 0
    for count in counts:
        new_bits = []
        for b in bits:
            new_bits.append(b + '0')
            new_bits.append(b + '1')
        for _ in range(count):
            bit = new_bits.pop(0)
            huff_tbl[bit] = f'{symbols[idx]:02X}' if hex else symbols[idx]
            idx += 1
        bits = new_bits
    if encoding:
        huff_tbl = {value: key for key, value in huff_tbl.items()}
    return huff_tbl

Y_DC_HUFF_TBL = expand_huffman_tbl(**y_dc)
Y_AC_HUFF_TBL = expand_huffman_tbl(**y_ac)
C_DC_HUFF_TBL = expand_huffman_tbl(**c_dc)
C_AC_HUFF_TBL = expand_huffman_tbl(**c_ac)
