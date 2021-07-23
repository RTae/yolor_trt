FROM nvcr.io/nvidia/tensorrt:21.06-py3

WORKDIR /usr/src/app

RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt==7.2.3.4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y ffmpeg libsm6 libxext6