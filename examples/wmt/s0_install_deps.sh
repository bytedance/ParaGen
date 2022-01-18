# install pycld3
pip install pycld3

# install sentencepiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd ../..

# download mosesdecoder
git clone https://github.com/moses-smt/mosesdecoder.git

