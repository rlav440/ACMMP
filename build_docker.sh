#!/bin/bash

docker build --no-cache -t accmp_docker -f docker/Dockerfile .


# # Save docker image to file 
# sudo docker save <image-name> > <image-name>.tar

# # Load docker image to file 
# sudo docker load < <image-name>.tar