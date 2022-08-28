#!/usr/bin/bash
RELEASE="3.4.11"
echo $RELEASE
FFMPEG_FOLDER="ffmpeg-$RELEASE"
FFMPEG_TAR="$FFMPEG_FOLDER.tar.gz"
FFMPEG_URL="https://ffmpeg.org/releases/$FFMPEG_TAR"
echo "Download and install ffmpg -> $FFMPEG_URL"
cd /tmp && wget $FFMPEG_URL &&\
    tar xvf $FFMPEG_TAR && \
    rm $FFMPEG_TAR && \
    cd $FFMPEG_FOLDER && \
    ./configure && \
    make && \
    make install