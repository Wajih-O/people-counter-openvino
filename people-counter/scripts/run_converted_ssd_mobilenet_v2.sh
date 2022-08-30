#!/usr/bin/bash
. ../.venv/bin/activate && python app.py infer --models_root_dir="./models/converted" --model="ssd_mobilenet_v2_coco_2018_03_29" --model_precision="default" | ffmpeg -v warning -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 24 -i - http://ffserver:3005/fac.ffm
