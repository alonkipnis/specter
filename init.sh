# for macos
export -u MACOSX_DEPLOYMENT_TARGET=10.15

wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
tar -xzvf archive.tar.gz 

pip3 install -r requirements.txt  
python setup.py install