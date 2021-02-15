
function install_jpeg2dct() {
    echo "Install jpeg2dct"
    git clone https://github.com/uber-research/jpeg2dct.git
    os=$(uname)
    if [ $os = "Darwin" ]
    then
        sed -i ''  s/\<string\>/\<tstring\>/g jpeg2dct/jpeg2dct/tensorflow/tf_lib.cc
        sed -i '' '52,59d' jpeg2dct/jpeg2dct/tensorflow/__init__.py
        sed -i '' '158,160d' jpeg2dct/jpeg2dct/common/dctfromjpg.cc  
    elif [ $os = "Linux" ]
    then
        sed -i s/\<string\>/\<tstring\>/g jpeg2dct/jpeg2dct/tensorflow/tf_lib.cc
        sed -i '52,59d' jpeg2dct/jpeg2dct/tensorflow/__init__.py
        sed -i '158,160d' jpeg2dct/jpeg2dct/common/dctfromjpg.cc  
    fi

    cd jpeg2dct && python3 setup.py install
    cd ..
    rm -rf jpeg2dct
}


install_jpeg2dct
echo 'Install pyyaml'
pip3 install PyYAML==5.3.1
echo 'Install PyTurboJPEG'
pip3 install -U git+git://github.com/lilohuang/PyTurboJPEG.git
