# this image installs python 3.9, pytorch and tensorflow all working with cuda 11.7 gpu acceleration
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# setup the lama dependencies
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Preconfigure tzdata
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections
RUN echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y sudo wget mc tmux nano build-essential rsync libgl1 git ca-certificates unzip zip dos2unix libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=user
RUN addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    USER=$USERNAME && \
    GROUP=$USERNAME
    
USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/miniconda3/bin:/home/$USERNAME/.local/bin:${PATH}"

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh && \
    echo "536817d1b14cb1ada88900f5be51ce0a5e042bae178b5550e62f61e223deae7c /tmp/miniconda.sh" > /tmp/miniconda.sh.sha256 && \
    sha256sum --check --status < /tmp/miniconda.sh.sha256 && \
    bash /tmp/miniconda.sh -bt -p "/home/$USERNAME/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda build purge && \
    conda init

RUN pip install -U pip
RUN pip install torch==2.0.1 torchvision==0.15.2
RUN pip install -U tensorflow==2.14.0
RUN pip install -U pip numpy scipy joblib matplotlib pandas pytorch-lightning==1.2.9 tabulate easydict==1.10 kornia==0.5.0 webdataset packaging gpustat tqdm pyyaml hydra-core==1.1.0 scikit-learn==1.3.1 tabulate
RUN pip install scikit-image==0.22.0 torchtext==0.6.0 albumentations==0.5.2 opencv-python-headless==4.8.1.78

# copy the concert data
#COPY concert /home/user/concert_data

# get the lama repo
ADD "https://api.github.com/repos/BabackE/lama_stereoinpainting/commits?per_page=1" latest_commit
RUN git clone https://github.com/BabackE/lama_stereoinpainting.git "/home/$USERNAME/lama"
ENV PYTHONPATH="/home/$USERNAME/lama"
ENV TORCH_HOME="/home/$USERNAME/lama"
ENV USER=$USERNAME

# this must be done after git is pulled to create the lama folder
RUN ln -s /home/user/concert_data /home/user/lama/concert
