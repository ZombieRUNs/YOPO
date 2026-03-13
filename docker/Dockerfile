# ============================================================
# YOPO: You Only Plan Once — build environment
# Base: CUDA 11.8 + Ubuntu 20.04 + ROS Noetic
# ============================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ---- 0. Use China mirrors for apt ----
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# ---- 1. System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl ca-certificates gnupg2 lsb-release \
    libzmqpp-dev libopencv-dev libpcl-dev libeigen3-dev \
    libboost-all-dev libflann-dev \
    libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa libx11-6 \
    python3-pip python3-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. ROS Noetic ----
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-roscpp \
    ros-noetic-tf \
    ros-noetic-nav-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-cv-bridge \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-image-transport \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=noetic
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ---- 3. Miniconda (Python 3.8 env) ----
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -y -n yopo python=3.8 && conda clean -ya
ENV CONDA_DEFAULT_ENV=yopo
ENV PATH=/opt/conda/envs/yopo/bin:$PATH

# ---- 4. Python dependencies (README spec) ----
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4" \
    && pip install --no-cache-dir \
    opencv-python \
    "gym==0.21.0" \
    "stable-baselines3==1.5.0" \
    "scipy==1.10.1" \
    "scikit-build==0.18.1" \
    "protobuf==3.20.3" \
    "PyYAML==6.0.1" \
    "ruamel.yaml==0.17.21" \
    "numpy==1.22.3" \
    "tensorboard==2.8.0" \
    empy \
    catkin_pkg \
    tqdm

# ---- 5. Environment variables ----
ENV FLIGHTMARE_PATH=/root/YOPO
ENV PYTHONPATH=${FLIGHTMARE_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ---- 6. Copy project & build flightlib ----
COPY . ${FLIGHTMARE_PATH}

WORKDIR ${FLIGHTMARE_PATH}/flightlib
RUN mkdir -p build && cd build \
    && source /opt/ros/noetic/setup.bash \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# flightgym.so needs to be on PYTHONPATH
ENV PYTHONPATH=${FLIGHTMARE_PATH}/flightlib/build:${PYTHONPATH}

# ---- 7. Make renderer executable ----
RUN chmod +x ${FLIGHTMARE_PATH}/flightrender/RPG_Flightmare/flightmare.x86_64 2>/dev/null || true

# ---- 8. Default workdir ----
WORKDIR ${FLIGHTMARE_PATH}/run

RUN cat <<'EOF' >> /root/.bashrc
# Prefer mounted workspace when available.
if [ -d /root/workspace/YOPO ]; then
  export FLIGHTMARE_PATH=/root/workspace/YOPO
else
  export FLIGHTMARE_PATH=/root/YOPO
fi
export PYTHONPATH="${FLIGHTMARE_PATH}:${FLIGHTMARE_PATH}/flightlib/build:${PYTHONPATH}"
EOF

RUN apt-get update && apt-get install -y --no-install-recommends \
    cpufrequtils \
    libompl-dev \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-rviz \
    ros-noetic-rqt-plot \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
