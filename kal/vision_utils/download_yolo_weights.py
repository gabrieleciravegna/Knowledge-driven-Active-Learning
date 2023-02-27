import os
import sys

import requests


def download(url, file_path):
    print("Downloading %s" % file_path)
    response = requests.get(url, allow_redirects=True, stream=True)
    total_length = response.headers.get('content-length')

    with open(file_path, "wb") as f:
        dl = 0
        total_length = int(total_length)
        for data in response.iter_content(chunk_size=4096):
            dl += len(data)
            f.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
            sys.stdout.flush()

    print("\n")


def download_yolo_weights(root: str, override=False):

    # Download weights for vanilla YOLOv3
    if not os.path.exists(os.path.join(root, "yolov3.weights")) or override:
        download("https://pjreddie.com/media/files/yolov3.weights", os.path.join(root, "yolov3.weights"))

    # Download weights for tiny YOLOv3
    if not os.path.exists(os.path.join(root, "yolov3-tiny.weights")) or override:
        download("https://pjreddie.com/media/files/yolov3-tiny.weights", os.path.join(root, "yolov3-tiny.weights"))

    # Download weights for backbone network
    if not os.path.exists(os.path.join(root, "darknet53.conv.74")) or override:
        download("https://pjreddie.com/media/files/darknet53.conv.74", os.path.join(root, "darknet53.conv.74"))

    return os.path.join(root, "darknet53.conv.74")


if __name__ == "__main__":
    download_yolo_weights(root=".")