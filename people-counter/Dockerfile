FROM openvino/ubuntu20_dev:latest
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/openvino
ENV WORKSPACE=/home/openvino/people-counter

USER root

RUN apt-get -yqq update \
    && apt-get install -yq --no-install-recommends \
      git \
      tree \
      yasm libx264-dev \
      vim \
      make \
      g++ \
      wget \
      npm \
      libzmq3-dev \
      libkrb5-dev \
      python3-tk python3-dev\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg with ffserver (compatible with ffserver service/container)
COPY scripts/install_ffmpeg_3.4.sh /tmp/
RUN /tmp/install_ffmpeg_3.4.sh

# nodejs npm
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash \
    && apt-get install -y nodejs \
    && npm -v \
    && npm update npm -g \
    && npm -v


USER openvino
WORKDIR $HOME

RUN touch ${HOME}/.Xauthority
RUN mkdir ${HOME}/scripts
COPY scripts/ ${HOME}/scripts

# create virtualenv and install requirements
COPY requirements* ${HOME}/
RUN tree ${HOME}/scripts
RUN cd ${HOME} && ${HOME}/scripts/create-env.sh

# TODO: install openvino-utils (library)
# TODO: export DISPLAY

WORKDIR $WORKSPACE

ENTRYPOINT ["tail", "-f", "/dev/null"]