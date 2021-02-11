!#/usr/bin/env

git clone https://github.com/uber-research/jpeg2dct.git
sed -i s/\<string\>/\<tstring\>/g jpeg2dct/jpeg2dct/tensorflow/tf_lib.cc
sed -i '52,59d' jpeg2dct/jpeg2dct/tensorflow/__init__.py
sed -i '158,160d' jpeg2dct/jpeg2dct/common/dctfromjpg.cc  # delete `transcode`
cd jpeg2dct && python3 setup.py install
cd ..

git clone https://github.com/dwgoon/jpegio.git
cd jpegio && python3 setup.py install
cd ..

pip install simplejpeg &> /dev/null
