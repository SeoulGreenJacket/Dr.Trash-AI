import pika
import json

host = "seheon.email"
port = 5672
queue_name = 'frame-data-queue'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port))
channel = connection.channel()
channel.queue_declare(queue=queue_name, durable=True)


def consume_frame_data(ch, method, properties, body):
    json_raw = body.decode()
    print(json_raw)
    frame_data = json.loads(json_raw)
    print(frame_data)
    # Something Amazing Important Perfectly Working Here
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(
    queue=queue_name, on_message_callback=consume_frame_data)
channel.start_consuming()
