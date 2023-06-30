
```bash

# install cuda 


# if oepnCVConfig.cmake file missing either run:
sudo find / -name "OpenCVConfig.cmake"
# or 
sudo apt-get install libopencv-dev # works better

# if missing boost paths then run:
sudo apt-get install libboost-all-dev


# CMakeFile.txt - Gencodes for nvida architectures
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

for cuda 12.1
-gencode arch=compute_60,code=[sm_60,compute_60] -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_75,code=[sm_75,compute_75] -gencode arch=compute_80,code=[sm_80,compute_80] -gencode arch=compute_86,code=[sm_86,compute_86] -gencode arch=compute_87,code=[sm_87,compute_87] -gencode arch=compute_89,code=[sm_89,compute_89] -gencode arch=compute_90,code=[sm_90,compute_90]

for cuda 11.4.1
-gencode arch=compute_60,code=[sm_60,compute_60] -gencode arch=compute_61,code=[sm_61,compute_61] -gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_75,code=[sm_75,compute_75] -gencode arch=compute_80,code=[sm_80,compute_80] -gencode arch=compute_86,code=[sm_86,compute_86] -gencode arch=compute_87,code=[sm_87,compute_87] 




# 

```