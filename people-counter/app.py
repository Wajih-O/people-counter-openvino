import argparse
import json
import os
import sys

from random import randint
from time import monotonic

import cv2
import numpy as np

import paho.mqtt.client as mqtt
from commandr import command, Run

from openvino_utils.utils import ImageDimension
from openvino_utils.video_utils import codec

from pedestrian_detection import PedestrianDetection


# MQTT server environment variables
MQTT_HOST = "mqtt"
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 60


def ensure_output_directory(output_directory: str):
    """Ensure output directory exists (todo: try to create it)"""
    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            raise Exception(f"{output_directory} is not a directory")
    else:
        os.makedirs(output_directory)


@command
def infer(
    models_root_dir="./models/intel",
    model="pedestrian-detection-adas-0002",
    model_precision="FP16",
    input_file="./data/Pedestrian_Detect_2_1_1.mp4",
    output_directory="./output/",
    frames_window=10,  # last frame(s) window to smooth detection per frame signal/time-series
    threshold=0.7,
):

    """Main function to infer video:detect persons in video frames, process it to extract the entry and leave the scene
    output augmented output frames to sys.stdout and publish detection statistics to MQTT.


    :param models_root_dir : root directory for the (xml/bin) models
    :param model: model name
    :param model_precision: model precision (default FP16)
    :param input_file: data source when the type is set to video
    :param output_directory: output directory for the generated artifacts control video capture and benchmarking data
    :param frames_window: last frame(s) window to smooth detection per frame signal/time-series

    """

    ensure_output_directory(output_directory=output_directory)

    # Connect to the MQTT server (using MQTT protocol: both MQTT and websocket are enabled in MQTT container)
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Get and open video capture
    cap = cv2.VideoCapture(input_file)
    cap.open(input_file)

    output_dimension = ImageDimension(x=1280, y=720).scale(1)
    pedestrian_detection = PedestrianDetection(
        model_name=model, model_directory=f"{models_root_dir}/{model}/{model_precision}"
    )
    pedestrian_detection.load_model()

    video_writer = cv2.VideoWriter(
        "pedestrian_detection_output.mp4",
        codec(),
        30,
        (output_dimension.width, output_dimension.height),
    )

    pedestrians_in_frame = [0]  # initialized (a time series for people in the frame)
    smoothed_pedestrians_in_frame = [0]  # averaged last (frames_window) frame

    start = monotonic()
    # last_frame = False

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output_frame = cv2.resize(
            image,
            (output_dimension.width, output_dimension.height),
        )
        pedestrians = pedestrian_detection.detect(image, min_confidence=0.85)[:1]
        # TODO: use Jaccard index to remove the non-maxima (instead of limiting the pedestrian to 1)
        #  -> that should lead to the same behavior (with a fairly high min confidence)

        pedestrians_in_frame.append(len(pedestrians))
        smoothed_pedestrians_in_frame.append(
            int(np.mean(pedestrians_in_frame[-frames_window:]) > threshold)
        )
        if smoothed_pedestrians_in_frame[-2] != smoothed_pedestrians_in_frame[-1]:
            if smoothed_pedestrians_in_frame[-1] == 1:
                start = monotonic()
                client.publish(
                    "person", json.dumps({"count": smoothed_pedestrians_in_frame[-1]})
                )
            else:  # assumes one person in the scene
                client.publish(
                    "person/duration", json.dumps({"duration": monotonic() - start})
                )

        for pedestrian in pedestrians:
            output_frame = cv2.resize(
                pedestrian.draw(output_frame),
                (output_dimension.width, output_dimension.height),
            )

        client.publish("person", json.dumps({"count": len(pedestrians)}))

        # write output_frame to an mp4 video output
        video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

        # Send frame to the stdout to pipe to ffmpeg server
        sys.stdout.buffer.write(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # release the output (video_writer)
    video_writer.release()
    # release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()

    # saving perf. statistics/summary
    perf_stats = {"precision": model_precision, "average_prediction_time": {}}
    if pedestrian_detection.prediction_time:
        perf_stats["average_prediction_time"][
            pedestrian_detection.model_name
        ] = np.mean(pedestrian_detection.prediction_time)
    with open(
        os.path.join(
            output_directory,
            f"perf_summary_{pedestrian_detection.model_name}_{model_precision}.json",
        ),
        "w",
    ) as perf_output:
        json.dump(perf_stats, perf_output)


if __name__ == "__main__":
    Run()
