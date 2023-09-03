FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA='cu121'

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3.10 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists && \
    ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip 
RUN python -m pip install --pre torch  --index-url https://download.pytorch.org/whl/nightly/$CUDA

# zsh plugin
ENV SHELL /bin/zsh
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" \
    && echo "$SNIPPET" >> "/root/.zsh_history"
COPY .devcontainer/zsh-plugins/.p10k.zsh /root/

# pip install
COPY .devcontainer/requirements.txt /workspace/requirements.txt
RUN  --mount=type=cache,target=/root/.cache/pip python -m pip install -r /workspace/requirements.txt

# install cmake
ARG CMAKE_VERSION=3.27.3
RUN curl -O -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    yes | sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# install nvidia nsight
RUN apt update && \
    apt install -y cuda-nsight-systems-12-2


    









