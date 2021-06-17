FROM continuumio/anaconda3:2019.03

# RUN pip install --upgrade pip && \
#     pip install autopep8 && \
#     pip install Keras && \
#     pip install tensorflow 

# WORKDIR /workdir
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
ADD environment.yml .
WORKDIR .
RUN echo 'PATH="$PATH:/path/to/pyenv"' >> ~/.bashrc
RUN conda init bash

RUN conda env create -f environment.yml
#pathを読み込んで処理

RUN echo "conda activate test" > ~/.bashrc
RUN source ~/.bashrc

RUN pip install -r requirements.txt