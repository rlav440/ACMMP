

### Install Nivida Proprietary Drivers 
```bash 
# --------------------------------------------------------
# https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu
# --------------------------------------------------------


# Update repo and packages 
sudo apt update && sudo apt upgrade -y 

# Find right drivers on apt repo
sudo apt search nvidia-driver

# Install Nvidia Driver 
sudo apt install <driver_name> -y # nvidia-driver-530

# Reboot 
sudo reboot 




# To uninstall nvidia drivers 
# --------

# check installed nvidia packages
dpkg -l | grep -i nvidia


# to remove nvidia drivers | ** IT MAY UNINSTALL DESKTOP ** 
sudo apt-get remove --purge '^nvidia-.*'


# if desktop uninstalled run:
sudo apt-get install ubuntu-desktop

# reboot 
sudo reboot 


```

### Install Nvidia CUDA
```bash 

# Follow instructuons on this link for this spacific version:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local



# After following above instructions set your path to point to CUDA binaries
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc



```

### Install Nvidia Container Tool Kit 

```bash 
# Install docker first then install nvidia container toolkit 



# Refer to Docs 
# ------------------
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Go directly to the Docker section to add package reposityy and install nvidia-container-toolkit





```