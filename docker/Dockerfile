# ============================================================
# YOPO: You Only Plan Once — build environment
# Base: CUDA 12.8 + Ubuntu 20.04 + ROS Noetic
# ============================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu20.04

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

# ---- 2.1 Gazebo Classic / Ignition packages for PX4 SITL ----
# Mirror PX4's ubuntu.sh behavior on Ubuntu 20.04 by enabling the OSRF repo.
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && wget -q http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-get update

ENV ROS_DISTRO=noetic
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ---- 3. Miniconda (Python 3.10 env for newer PyTorch/Blackwell support) ----
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -y -n yopo python=3.10 \
    && conda create -y -n px4 python=3.10 \
    && conda clean -ya
ENV CONDA_DEFAULT_ENV=yopo
ENV PATH=/opt/conda/envs/yopo/bin:$PATH

# ---- 4. YOPO Python dependencies ----
# PyTorch 2.7.x + CUDA 12.8 is the first official wheel line with Blackwell support.
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4" \
    && pip install --no-cache-dir \
    opencv-python \
    "einops>=0.7" \
    "omegaconf>=2.3" \
    "hydra-core>=1.3" \
    imageio \
    "gym==0.21.0" \
    "stable-baselines3==1.5.0" \
    "scipy==1.10.1" \
    "scikit-build==0.18.1" \
    "protobuf==3.20.3" \
    "PyYAML==6.0.1" \
    "ruamel.yaml==0.17.21" \
    "numpy==1.22.3" \
    "tensorboard==2.8.0" \
    "empy==3.3.4" \
    catkin_pkg \
    tqdm

# ---- 4.1 PX4 Python dependencies (isolated env to avoid polluting YOPO) ----
# PX4's upstream requirements.txt still contains the old pip-incompatible spec
# `matplotlib>=3.0.*`, so install the equivalent cleaned dependency set here.
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda run -n px4 python -m pip install --no-cache-dir "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4" \
    && conda run -n px4 python -m pip install --no-cache-dir \
    argcomplete \
    cerberus \
    coverage \
    "empy==3.3.4" \
    future \
    "jinja2>=2.8" \
    jsonschema \
    kconfiglib \
    lxml \
    "matplotlib>=3.0" \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "nunavut>=1.1.0" \
    packaging \
    "pandas>=0.21" \
    pkgconfig \
    psutil \
    pygments \
    pymavlink \
    pyros-genmsg \
    pyserial \
    "pyulog>=0.5.0" \
    pyyaml \
    requests \
    "six>=1.12.0" \
    "toml>=0.9" \
    "sympy>=1.10.1" \
    "symforce>=0.9.0"

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
export PX4_AUTOPILOT_PATH=/root/workspace/PX4-Autopilot
source /opt/conda/etc/profile.d/conda.sh
alias activate_yopo='conda activate yopo'
alias activate_px4='conda activate px4'

# PX4 Gazebo Classic environment
if [ -d "${PX4_AUTOPILOT_PATH}" ]; then
  export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH}:${PX4_AUTOPILOT_PATH}"
  export ROS_PACKAGE_PATH="${ROS_PACKAGE_PATH}:${PX4_AUTOPILOT_PATH}/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
  if [ -d "${PX4_AUTOPILOT_PATH}/build/px4_sitl_default" ]; then
    source "${PX4_AUTOPILOT_PATH}/Tools/simulation/gazebo-classic/setup_gazebo.bash" \
      "${PX4_AUTOPILOT_PATH}" \
      "${PX4_AUTOPILOT_PATH}/build/px4_sitl_default"
  fi
fi
EOF

RUN apt-get update && apt-get install -y --no-install-recommends \
    cpufrequtils \
    libompl-dev \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-rviz \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-plugins \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-rqt \
    ros-noetic-rqt-graph \
    ros-noetic-rqt-reconfigure \
    ros-noetic-rqt-plot \
    astyle \
    cppcheck \
    file \
    gdb \
    lcov \
    libfuse2 \
    libxml2-dev \
    libxml2-utils \
    ninja-build \
    rsync \
    shellcheck \
    unzip \
    zip \
    ant \
    openjdk-13-jre \
    openjdk-13-jdk \
    libvecmath-java \
    bc \
    dmidecode \
    geographiclib-tools \
    gazebo11 \
    gazebo11-common \
    gazebo11-plugin-base \
    libgazebo11-dev \
    libignition-common3 \
    libignition-common3-dev \
    libignition-common3-graphics \
    libignition-common3-graphics-dev \
    libignition-math6 \
    libignition-math6-dev \
    libignition-msgs5 \
    libignition-msgs5-dev \
    libignition-fuel-tools4 \
    libignition-fuel-tools4-dev \
    libignition-transport8 \
    libignition-transport8-dev \
    libignition-tools-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libimage-exiftool-perl \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7.1.0 /opt/conda/envs/yopo/lib/libffi.so.7

# Ensure ROS Python runtime deps are available in the active conda env.
RUN pip install --no-cache-dir rospkg catkin_pkg

# PX4 helper env stays isolated in its own conda environment.
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda run -n px4 python -m pip install --no-cache-dir rospkg catkin_pkg

# Pre-install GeographicLib datasets so MAVROS can start without manual setup.
RUN /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh

# Install OSQP system-wide for OMMPC.
RUN git clone --recursive -b release-0.6.3 https://github.com/osqp/osqp.git /tmp/osqp \
    && cmake -S /tmp/osqp -B /tmp/osqp/build -G "Unix Makefiles" \
    && cmake --build /tmp/osqp/build \
    && cmake --build /tmp/osqp/build --target install \
    && ldconfig \
    && rm -rf /tmp/osqp

# PX4 source lives on the mounted workspace at runtime; trust these repos for root.
RUN git config --global --add safe.directory /root/workspace/PX4-Autopilot \
    && git config --global --add safe.directory /root/workspace/PX4-Autopilot/src/modules/mavlink/mavlink \
    && git config --global --add safe.directory /root/workspace/PX4-Autopilot/platforms/nuttx/NuttX/nuttx \
    && git config --global --add safe.directory /root/workspace/PX4-Autopilot/platforms/nuttx/NuttX/apps

CMD ["/bin/bash"]
