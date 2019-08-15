# FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
FROM ubuntu:16.04
MAINTAINER Christian Schroeder de Witt

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

#Install python3 pip3
RUN apt-get update
RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.6 python3.6-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN pip3 install --upgrade pip
RUN apt-get install -y python-apt --reinstall
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger

RUN pip install numpy
RUN pip install torch -f https://download.pytorch.org/whl/nightly/cpu/torch.html

RUN apt-get install -y libhdf5-serial-dev netcdf-bin libnetcdf-dev
#RUN apt-get install -y python3-netcdf
RUN pip3 install netcdf4
