#!/bin/sh
conda create -n ipcl python=3.6 mamba -c conda-forge -y
source activate ipcl

conda config --env --add channels anaconda 
conda config --env --add channels conda-forge

mamba uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
mamba install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
mamba install -y jpeg libtiff
mamba install -c conda-forge packaging
mamba install pytorch=0.4.1 cuda92 -c pytorch
mamba install -c anaconda pytz -y
mamba install -c fastai fastprogress -y
mamba install -c menpo opencv -y
mamba install -c conda-forge qt -y
mamba install -c conda-forge albumentations -y
mamba install -c anaconda psutil
mamba install -c anaconda joblib
mamba install -c conda-forge kornia

mamba install ipykernel -y
ipython kernel install --user --name=ipcl

pip install -U addict==2.3.0 --user
pip install kornia==0.2.0 --user
pip install pyrtools --user
pip install git+https://github.com/rwightman/pytorch-image-models.git -U

conda deactivate