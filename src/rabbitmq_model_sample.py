import pika
import json
import base64

host = "seheon.email"
port = 5672
frame_image_queue_name = 'frame-image-queue'
frame_data_queue_name = 'frame-data-queue'

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host, port=port))
frame_image_channel = connection.channel()
frame_image_channel.queue_declare(queue=frame_image_queue_name, durable=True)
frame_data_channel = connection.channel()
frame_data_channel.queue_declare(queue=frame_data_queue_name, durable=True)


def image_to_objects(image):
    objects = []
    # Something PERFECTLY AMAZING IMPORTANT WORKING with image HERE
    return objects


def consume_frame_image(ch, method, properties, body):
    frame = json.loads(body.decode())
    timestamp, cameraId, image = frame['timestamp'], frame['cameraId'], frame['image']
    image = base64.b64decode(image)
    objects = image_to_objects(image)
    data = {'timestamp': timestamp, 'cameraId': cameraId, 'objects': objects}
    message = json.dumps(data)
    frame_data_channel.basic_publish(
        exchange='', routing_key=frame_data_queue_name, body=message, properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE))
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    frame_image_channel.basic_qos(prefetch_count=1)
    frame_image_channel.basic_consume(
        queue=frame_image_queue_name, on_message_callback=consume_frame_image)
    frame_image_channel.start_consuming()


if __name__ == '__main__':
    main()
