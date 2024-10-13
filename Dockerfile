# 使用Ubuntu 20.04基础镜像
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
#registry.cn-shenzhen.aliyuncs.com/hanyangdev/gaussiandev:v5.0

# 设置工作目录
WORKDIR /app

USER root

# Install some basic utilities
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y \
    tzdata \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    tmux \
    wget \
    zsh \
    vim \
    libssl-dev \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    openssh-server \
    openssh-client \
    iputils-ping \
    unzip \
    cmake \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
    pciutils \
    xauth \
    llvm \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    build-essential \
    locales \
    python-is-python3 \
    python3.8 \
    python3-pip \
    && apt-get clean \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# （必要）修改SSH配置
RUN mkdir -p /var/run/sshd && \
    sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config 

# （可选，但强烈推荐）安装Jupyterlab
RUN pip install --no-cache-dir --upgrade \
    jupyterlab>=3.0.0 \
    ipywidgets \
    matplotlib \
    jupyterlab_language_pack_zh_CN \
    -i https://mirrors.aliyun.com/pypi/simple


#（可选）设置语言和时区
RUN export DEBIAN_FRONTEND=noninteractive && \
    locale-gen zh_CN zh_CN.GB18030 zh_CN.GBK zh_CN.UTF-8 en_US.UTF-8 && \
    update-locale && \
    echo "LANG=en_US.UTF-8" >> /etc/profile && \
    echo "LANGUAGE=en_US:en" >> /etc/profile && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/profile && \
    cp -f /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo 'Asia/Shanghai' >/etc/timezone


RUN pip3 install torch==2.0.1 torchvision -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
#--index-url https://download.pytorch.org/whl/cu117
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY IsaacGym_Preview_4_Package.tar.gz /app
# source activate gfhoi && \  
RUN /bin/bash -c "tar -xzf /app/IsaacGym_Preview_4_Package.tar.gz -C /app && \
    cd /app/isaacgym/python && \
    pip install -e ."

# 默认运行命令
CMD ["bash"]
