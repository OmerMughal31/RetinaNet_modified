# Define specific TF Version for compability
ARG TENSORFLOW_VERSION=1.14.0

# Parent Image is the offical TF image with Pyhton 3, GPU and Jupyter support
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}-gpu-py3-jupyter

# Create a non-root user with name "containeruser" and add sudo privileges
# This user gets used when you connect to a Docker Container from VSC. 
# See https://aka.ms/remote/containers-advanced#_creating-a-nonroot-user for details.
ARG USERNAME=containeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the working directory
WORKDIR /tf
RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev

# Copy requrements.txt into Docker image and install python dependencies with pip
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
