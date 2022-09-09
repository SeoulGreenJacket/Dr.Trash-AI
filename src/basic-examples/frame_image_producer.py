import pika
import json
import base64

host = "seheon.email"
port = 5672
queue_name = 'frame-image-queue'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port))
channel = connection.channel()

channel.queue_declare(queue=queue_name, durable=True)

image_raw = b"raw binary data"
image_base64 = base64.b64encode(image_raw)
data = {'timestamp': 1234567890, 'cameraId': 1, 'image': image_base64}
message = json.dumps(data)
channel.basic_publish(exchange='', routing_key=queue_name, body=message,
                      properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE))
connection.close()
