"""Kombu-based Video Stream Consumer

Written by Minsu Jang
Date: 2018-06-09

Reference
- Building Robust RabbitMQ Consumers With Python and Kombu: Part 1 (https://medium.com/python-pandemonium/building-robust-rabbitmq-consumers-with-python-and-kombu-part-1-ccd660d17271)
- Building Robust RabbitMQ Consumers With Python and Kombu: Part 2 (https://medium.com/python-pandemonium/building-robust-rabbitmq-consumers-with-python-and-kombu-part-2-e9505f56e12e)
"""

import cv2
import numpy as np
import sys
import time

from kombu import Connection, Exchange, Queue
from kombu.mixins import ConsumerMixin

from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle

import json
import os
from collections import defaultdict
from pathlib import Path

from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import eval

# Default RabbitMQ server URI
rabbit_url = 'amqp://guest:guest@localhost:5672//'
global args
class fix_parameters:
    def __init__(self):
        self.ap_data_file='results/ap_data.pkl'
        self.bbox_det_file='results/bbox_detections.json'
        self.benchmark=False
        self.config=None
        self.crop=True
        self.cross_class_nms=False
        self.cuda=True
        self.dataset=None
        self.detect=False
        self.display=False
        self.display_bboxes=True
        self.display_fps=False
        self.display_lincomb=False
        self.display_masks=True
        self.display_scores=True
        self.display_text=True
        self.emulate_playback=False
        self.fast_nms=True
        self.image=None
        self.images=None
        self.mask_det_file='results/mask_detections.json'
        self.mask_proto_debug=False
        self.max_images=-1
        self.no_bar=False
        self.no_hash=False
        self.no_sort=False
        self.output_coco_json=False
        self.output_web_json=False
        self.resume=False
        self.score_threshold=0.6
        self.seed=None
        self.shuffle=False
        self.top_k=15
        self.trained_model='weights/yolact_base_54_800000.pth'
        self.video='0'
        self.video_multiframe=4
        self.web_det_path='web/dets/'

def evalimage(net:Yolact, image=np.array):
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy = eval.prep_display(preds, frame, None, None, undo_transform=False)
    return img_numpy

def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    return evalimage(net, args.image)

# Kombu Message Consuming Worker
class Worker(ConsumerMixin):
    def __init__(self, connection, queues):
        self.connection = connection
        self.queues = queues

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=self.queues,
                         callbacks=[self.on_message],
                         accept=['image/jpeg'])]

    def on_message(self, body, message):
        
        # get the original jpeg byte array size
        size = sys.getsizeof(body) - 33
        # jpeg-encoded byte array into numpy array
        np_array = np.frombuffer(body, dtype=np.uint8)
        np_array = np_array.reshape((size, 1))
        # decode jpeg-encoded numpy array 
        image = cv2.imdecode(np_array, 1)

        args.image = image
        image = evaluate(net, dataset)


        # show image
        cv2.imshow("image", image)
        cv2.waitKey(1)

        # send message ack
        message.ack()

def run():
    exchange = Exchange("video-exchange", type="direct")
    queues = [Queue("video-queue", exchange, routing_key="video")]
    with Connection(rabbit_url, heartbeat=4) as conn:
            worker = Worker(conn, queues)
            worker.run()

if __name__ == "__main__":
    args = fix_parameters()
    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()
    
    
        run()
