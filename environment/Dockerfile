# hash:sha256:d8f48661183a4f744a312160ed0e1bddf7f668658c77409c41237e12950c7d04
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    clip==0.2.0 \
    scikit-image==0.25.2 \
    ultralytics==8.0.120
