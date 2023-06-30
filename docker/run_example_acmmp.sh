#!/bin/bash

# To build image from Dockerfile
# ------------------------------- 
# docker build -t test/nvidia .
# docker build --no-cache -t acmmp_build_1804/nvidia .  | Image tag has to be lowercase


# docker run --rm -it -P \
#     --name acmmp_test \
#     -v /home/rlav440/CLionProjects/ACMMP/recons/sketch/scan33_9_cam:/acmmp_dir \
#     --runtime=nvidia \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     -e CUDA_INCLUDE_DIRS=/usr/local/cuda/include \
#     -e CUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so \
#     --gpus all \
#     acmmp_build_test/nvidia


docker run --rm -it --rm \
    --name acmmp_nvidia \
    -v /mnt/nvme_fast/flir_capture/acmmp_recon_robin_ray_kiwi_20230628/acmmp_hand_ray_20230628_181215/acmmp_output_4:/acmmp_dir \
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    acmmp_cuda-12.1.0/nvidia



# to run after 
# docker exec -it test_nvidia_container /bin/bash
