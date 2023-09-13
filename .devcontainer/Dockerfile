FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# ARG CUDA='cu121'

RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   wget \
                   zsh \
                   ca-certificates 
#                    python3.10 \
#                    python3-pip && \
#     ln -s /usr/bin/python3.10 /usr/bin/python
# RUN python -m pip install --no-cache-dir --upgrade pip 
# RUN python -m pip install --pre torch  --index-url https://download.pytorch.org/whl/nightly/$CUDA

# instal conda 
RUN mkdir ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh 

COPY .devcontainer/environment.yml /workspace/environment.yml
RUN ~/miniconda3/bin/conda env create -f /workspace/environment.yml

# zsh plugin
ENV SHELL /bin/zsh
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" \
    && echo "$SNIPPET" >> "/root/.zsh_history"
COPY .devcontainer/zsh-plugins/.p10k.zsh /root/

# # pip install
# COPY .devcontainer/requirements.txt /workspace/requirements.txt
# RUN  --mount=type=cache,target=/root/.cache/pip python -m pip install -r /workspace/requirements.txt

# install cmake
ARG CMAKE_VERSION=3.27.3
RUN curl -O -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    yes | sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# install nvidia nsight
COPY .devcontainer/systems.deb /workspace/systems.deb
RUN --mount=type=cache,target=/var/cache/apt \
    apt install -y  libglib2.0-0 \
                    libtinfo5 \
                    libxcb-xinerama0 \
                    libxcb-icccm4 \
                    libxcb-image0 \
                    libxcb-keysyms1 \
                    libxcb-randr0 \
                    libxcb-render-util0 \
                    libxcb-xfixes0 \
                    libxcb-shape0 \
                    libxkbcommon-x11-0 \
                    libxcb-xinput0 \
                    libfontconfig1 \
                    libdbus-1-3 \
                    libnss3 \
                    libxcomposite1 \
                    libxdamage1 \
                    libxrandr2 \
                    libxtst6 \
                    libasound2 \
                    libopengl0 \
                    libegl1 && \
    dpkg -i /workspace/systems.deb && \
    rm /workspace/systems.deb


# set cuda env in .zshrc
RUN echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> /root/.zshrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /root/.zshrc && \
    echo 'export CUDA_HOME=/usr/local/cuda' >> /root/.zshrc