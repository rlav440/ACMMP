# devel needed to have nvcc inside container
# ------------------------------------------
# FROM nvidia/cuda:12.1.1-devel-ubuntu20.04 
FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

# set working directory inside container
WORKDIR /ACMMP_docker_build

# copy over ACMMP build dir to container
COPY /src /ACMMP_docker_build

# ensures apt installs are non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Miscellaneous Components & libopencv & libboost
RUN apt-get update --fix-missing && \
apt-get install -y --no-install-recommends \
apt-utils git build-essential cmake libopencv-dev libboost-all-dev

# Complile ACMMP
RUN cmake /ACMMP_docker_build
RUN make -j$(nproc)

# cleans up after install to reduce size of container
RUN rm -rf /var/lib/apt/lists/* && \
apt-get remove -y apt-utils git build-essential cmake && \
apt-get autoremove -y


CMD ["/ACMMP_docker_build/ACMMP", "/acmmp_dir/"]
