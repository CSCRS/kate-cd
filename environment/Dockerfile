# hash:sha256:30b61073907c6059ec6b5d95ca8208a0325b6304a7e23d821aa32f5741d6e122
FROM registry.codeocean.com/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    clip==0.2.0 \
    scikit-image==0.25.2 \
    ultralytics==8.0.120
