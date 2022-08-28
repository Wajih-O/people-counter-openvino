FROM alpine:3.8
RUN apk add curl bash ffmpeg && \
    rm -rf /var/cache/apk/*

COPY server.conf /etc/ffserver.conf

ENTRYPOINT [ "ffserver" ]
