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
#     --gpus all \
#     acmmp/full:cuda-11.4.0-devel

# docker run --rm -it \
#     --name acmmp_nvidia \
#     -v /mnt/nvme_fast/flir_capture/acmmp_recon_robin_ray_kiwi_20230628/acmmp_hand_ray_20230628_181215/acmmp_output_4:/acmmp_dir \
#     --runtime=nvidia \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     --gpus all \
#     acmmp/full:cuda-11.4.0-devel

docker run --rm -it -P \
    --name acmmp_nvidia \
    -v /mnt/nvme_fast/acmmp_pipeline_test/acmmp_recon_mr_bones_20230704_160154/acmmp_output:/acmmp_dir \
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    acmmp/full:cuda-11.4.0-devel

# # FOR DEBUGING INSIDE CONTAINER 
# docker run --rm -it  \
#     --name acmmp_nvidia \
#     -p 8870:22 \
#     --runtime=nvidia \
#     -e NVIDIA_DRIVER_CAPABILITIES=all \
#     --gpus all \
#     acmmp/pipeline

docker run --rm -it \
    --name acmmp_nvidia \
    -v /mnt/nvme_fast/acmmp_pipeline_test/acmmp_hand_robin_20230628_182542/acmmp_output:/acmmp_output
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    acmmp/full:cuda-11.4.0-devel \
    --acmmp_output_dir /acmmp_output
    
docker run -d \
    --name acmmp_nvidia_debug \
    -p 7500:22 \
    --runtime=nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    acmmp/full:cuda-11.4.0-devel



# to run after 
# docker exec -it test_nvidia_container /bin/bash
