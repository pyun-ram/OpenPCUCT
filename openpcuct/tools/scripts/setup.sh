ROOT_DIR=$1

cd $ROOT_DIR/../det3/ops/src/
python3 setup.py build && python3 setup.py install
echo "export PYTHONPATH=${PYTHONPATH}:/usr/app/" >> ~/.bashrc

cd $ROOT_DIR/pcuct/ops/jiou/
mkdir include
cd include/
git clone https://gitlab.com/libeigen/eigen.git -b 3.4
cd $ROOT_DIR
python setup_pcuct.py develop
cd $ROOT_DIR
python setup.py develop

pip install matplotlib pandas xlsxwriter
