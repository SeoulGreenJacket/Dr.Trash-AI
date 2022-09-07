import pika
import json

host = "seheon.email"
port = 5672
queue_name = 'frame-data-queue'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port))
channel = connection.channel()

channel.queue_declare(queue=queue_name, durable=True)

data = {'timestamp': 1234567890, 'cameraId': 1, 'objects': [
    {'type': 'plastic', 'verticies': [12, 34, 56, 78]}]}
message = json.dumps(data)
channel.basic_publish(exchange='', routing_key=queue_name, body=message,
                      properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE))
connection.close()
