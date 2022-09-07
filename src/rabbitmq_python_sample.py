import pika
import json
from transmit_to_database import trash

host = "seheon.email"
port = 5672
queue_name = 'frame-data-queue'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port))
channel = connection.channel()
channel.queue_declare(queue=queue_name, durable=True)


def check_trash(camera_id: int, objects):
    logs = []
    for object in objects:
        if object:
            # If something trashed
            logs.append({'trashcan_id': camera_id,
                         'type': 'plastic', 'ok': True})
    return logs


def consume_frame_data(ch, method, properties, body):
    frame_data = json.loads(body.decode())
    camera_id, objects = frame_data['cameraId'], frame_data['objects']
    logs = check_trash(camera_id, objects)
    for log in logs:
        trash(log['trashcan_id'], log['type'], log['ok'])
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(
    queue=queue_name, on_message_callback=consume_frame_data)
channel.start_consuming()
