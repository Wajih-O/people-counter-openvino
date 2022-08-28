import argparse
import json
import socket
import sys

from random import randint
from time import monotonic

import cv2

# import numpy as np

import paho.mqtt.client as mqtt

# from openvino_utils.input_feeder import InputFeeder
from openvino_utils.utils import ImageDimension
from openvino_utils.video_utils import codec

from pedestrian_detection import PedestrianDetection

INPUT_STREAM = "./data/Pedestrian_Detect_2_1_1.mp4"


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 60


def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def infer_on_video(args, model="pedestrian-detection-adas-0002", precision="FP32"):

    # Connect to the MQTT server (using MQTT protocol: both MQTT and websocket are enabled in MQTT container)
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    output_dimension = ImageDimension(x=1280, y=720).scale(1)
    pedestrian_detection = PedestrianDetection(
        model_directory=f"./models/intel/{model}/{precision}"
    )
    pedestrian_detection.load_model()

    video_writer = cv2.VideoWriter(
        "pedestrian_detection_output.mp4",
        codec(),
        30,
        (output_dimension.width, output_dimension.height),
    )

    pedestrians_in_frame = [
        0
    ]  # initialized (a time series for People/Pedestrian count)

    start = monotonic()
    last_frame = False

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
        pedestrians = pedestrian_detection.detect(image)

        if len(pedestrians) != pedestrians_in_frame[-1]:
            if pedestrians_in_frame[-1] == 0:
                start = monotonic()

            if pedestrians_in_frame[-1] > 0:  # assumes one person in the scene
                client.publish(
                    "person/duration", json.dumps({"duration": monotonic() - start})
                )

        if len(pedestrians):
            client.publish(
                "person/duration", json.dumps({"duration": monotonic() - start})
            )  # continuously updated person duration in the scene

        pedestrians_in_frame.append(len(pedestrians))
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


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
