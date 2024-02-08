#!/bin/bash

# ------------ Extra Commands -----------
# Save docker image to file 
# sudo docker save <image-name> > <image-name>.tar
# 
# Load docker image to file 
# sudo docker load < <image-name>.tar
# ---------------------------------------

# move to parent dir to give dockerfile access to all files inside docker
cd ..

# build command
# docker build --no-cache -t acmmp/feature-dev:cuda-12.1.1-devel -f docker/Dockerfile .
# docker build --no-cache -t acmmp/feature-dev:cuda-11.4.0-devel -f docker/Dockerfile .

docker build --no-cache -t acmmp/full_local:cuda-11.3.1-devel -f docker/Dockerfile .

