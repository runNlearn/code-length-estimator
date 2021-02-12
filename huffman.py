import yaml

from PIL import Image


__all__ = [
    'Y_DC_HUFF_TBL',
    'Y_AC_HUFF_TBL',
    'C_DC_HUFF_TBL',
    'C_AC_HUFF_TBL',
]


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


with open('huffman_table.yaml', 'r') as f:
    _tables = yaml.load(f, Loader=yaml.FullLoader)

Y_DC_HUFF_TBL = expand_huffman_tbl(**_tables['ydc'])
Y_AC_HUFF_TBL = expand_huffman_tbl(**_tables['yac'])
C_DC_HUFF_TBL = expand_huffman_tbl(**_tables['cdc'])
C_AC_HUFF_TBL = expand_huffman_tbl(**_tables['cac'])
