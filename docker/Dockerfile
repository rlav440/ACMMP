# ----------------------------------
# Dockerfile For Full ACMMP Build Image
# NOTES: cuda-XX.X.X-devel is needed to have nvcc inside container ** 
# ----------------------------------


# FROM acmmp/full:cuda-11.4.0-devel
# FROM acmmp/feature-dev:cuda-11.4.0-devel
FROM acmmp/base:cuda-11.4.0-devel

# Set working directory inside container from which other commands will run at
WORKDIR /ACMMP_docker_build

# Copy files over ACMMP src to container under /ACMMP_docker_build
COPY /src /ACMMP_docker_build
COPY /docker/acmmp_docker.py /acmmp_docker.py

# Ensures apt installs are non-interactive
ENV DEBIAN_FRONTEND=noninteractive


# # --------------SSH DEBUG----------------
# # Install SSH and create password
# RUN apt-get install -y openssh-server 
# RUN mkdir /var/run/sshd
# RUN echo 'root:password' | chpasswd
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# # SSH login fix. Otherwise user is kicked off after login
# RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# ENV NOTVISIBLE "in users profile"
# RUN echo "export VISIBLE=now" >> /etc/profile

# # Create User and add user to Sudoers 
# RUN useradd -ms /bin/bash  ubuntu
# RUN echo 'ubuntu:password' | chpasswd
# RUN echo "ubuntu 	ALL=(ALL:ALL) ALL" | tee -a /etc/sudoers >/dev/null
# # ------------------------------------



# Complile ACMMP
RUN cmake /ACMMP_docker_build
RUN make -j$(nproc)


# Cleans up after install to reduce size of container
RUN rm -rf /var/lib/apt/lists/* && \
apt-get remove -y apt-utils git build-essential cmake && \
apt-get autoremove -y



# Additional command to execute the binary with additional flags and an external file path
ENTRYPOINT ["python3", "/acmmp_docker.py"]

# Default arguments for the entrypoint command
CMD ["/acmmp_dir" ]


# # DEBUG
# EXPOSE 22
# CMD ["/usr/sbin/sshd", "-D"]
