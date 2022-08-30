#!/usr/bin/bash
. ../.venv/bin/activate && python app.py infer --model="pedestrian-detection-adas-0002"| ffmpeg -v warning -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 24 -i - http://ffserver:3005/fac.ffm
