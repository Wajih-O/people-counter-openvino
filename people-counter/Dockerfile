FROM openvino/ubuntu20_dev:latest
ENV DEBIAN_FRONTEND=noninteractive


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


# Set user and group
ARG user=openvino
ARG DISPLAY

# export DISPLAY (needed by openvino-utils library)
ENV DISPLAY=${DISPLAY}

# Switch to user
USER ${user}

ENV HOME=/home/${user}
ENV WORKSPACE=/home/${user}/people-counter

RUN touch ${HOME}/.Xauthority
RUN mkdir ${HOME}/scripts
COPY scripts/ ${HOME}/scripts

# create virtualenv and install requirements
WORKDIR $HOME
RUN pwd
COPY requirements* ${HOME}/
RUN tree ${HOME}/scripts
RUN ${HOME}/scripts/create-env.sh

# Install openvino-utils (a helper library)
RUN mkdir ${HOME}/openvino-utils
COPY openvino-utils/ ${HOME}/openvino-utils
RUN ${HOME}/scripts/pip_install.sh ${HOME}/openvino-utils


WORKDIR $WORKSPACE

# UI as entry point
ENTRYPOINT scripts/start_ui.sh

# DEBUG entry point
# ENTRYPOINT ["tail", "-f", "/dev/null"]