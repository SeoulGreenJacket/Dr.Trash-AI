from kafka import KafkaConsumer
import cv2 as cv
import numpy as np

consumer = KafkaConsumer('camera_000', bootstrap_servers='58.148.71.123:29092')
for bytes in consumer:
    buf = np.fromstring(bytes.value, np.uint8)
    frame = cv.imdecode(buf)
    print(type(frame))
