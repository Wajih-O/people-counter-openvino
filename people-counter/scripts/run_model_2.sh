#!/usr/bin/bash
export DISPLAY=172.25.16.1:0.0 && . ../.venv/bin/activate && python app.py infer --model="person-detection-0201" | ffmpeg -v warning -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 24 -i - http://ffserver:3005/fac.ffm
