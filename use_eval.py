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
import cv2
import eval
from collections import deque
import random
import string
from sklearn.utils.linear_assignment_ import linear_assignment

global trackers
trackers = []

def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

class PersonTracker(object):
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=30)
        self.large_roi = False
        self.small_roi = False
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

    def set_class(self, _class):
        self.cls_num = _class
        self.cls = cfg.dataset.class_names[_class]
        print(self.cls)
    '''
    def annotate(self, image):
        x1, y1, x2, y2 = self.bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        image = cv2.putText(image, self.activity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        image = cv2.drawMarker(image, self.centroid, (255, 0, 0), 0, 30, 4)
        return image
    '''

def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def tracker_match(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)
            #print('IOU_mat',IOU_mat[t,d])

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)
    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def trash_count(img, tracker):
    lx=0; ly=0; lw=500; lh=500
    large_roi = img[ly:ly+lh, lx:lx+lw]
    cv2.rectangle(large_roi, (0,0), (lh-1, lw-1), (0, 255, 0))

    sx=200; sy=200; sw=300; sh=300
    small_roi = img[sy:sy+sh, sx:sx+sw]
    cv2.rectangle(small_roi, (0,0), (sh-1, sw-1), (0, 0, 255))
    if tracker.h / lh > 0.8 or tracker.w / lw > 0.8:
        tracker.large_roi = True
    if tracker.large_roi is True:
        if sx < tracker.bbox[0] and sy < tracker.bbox[1] and sx+sw > tracker.bbox[2] and sy+sh > tracker.bbox[3]:
            tracker.small_roi = True

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

global args
args = fix_parameters()
args.trained_model = 'weights/yolact_resnet50_cig_trash_4210_80000.pth'
def evalimage(net:Yolact, image:np.array):
    frame = torch.from_numpy(image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_numpy, boxes, classes = eval.prep_display(preds, frame, None, None, undo_transform=False)
    return img_numpy, boxes, classes

def evaluate(net: Yolact, dataset,args=args, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    return evalimage(net, args.image)


import time
if __name__ == '__main__':
    args.trained_model='weights/yolact_resnet50_cig_trash_2631_50000.pth'
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
        cap = cv2.VideoCapture(0)
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상의 넓이(가로) 프레임
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 영상의 높이(세로) 프레임

        frame_size = (frameWidth, frameHeight)
        frameRate = 33
        while True:
            retval, img = cap.read()
            if not(retval):
                break

            args.image = img
            start = time.time()
            try:
                img, boxes, classes = evaluate(net, dataset, args=args)
            except:
                img = img
                boxes = []
                classes = []
            bboxes = []
            for j in range(len(boxes)):
                bboxes.append([classes[j], boxes[j]])
            track_boxes = [tracker.bbox for tracker in trackers]
            matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[1] for b in bboxes])

            for idx, jdx in matched:
                trackers[idx].set_class(bboxes[jdx][0])
                trackers[idx].set_bbox(bboxes[jdx][1])
            checkcheck = False  ##
            for tracker in trackers:
                print(tracker.large_roi)
                print(tracker.small_roi)
                print('\n')
            for idx in unmatched_detections:
                try:
                    ### --custom---
                    if trackers[idx].large_roi is True and trackers[idx].small_roi is True:
                        print('Congratulations Count {}'.format(trackers[idx].cls))
                        checkcheck = True
                    trackers.pop(idx)
                except:
                    pass
            if checkcheck is True:
                raise Exception('OK')  ##edit

            for idx in unmatched_trackers:
                person = PersonTracker()
                person.set_class(bboxes[idx][0])
                person.set_bbox(bboxes[idx][1])
                trackers.append(person)

            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1
            text_color = [255, 255, 255]
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes[j]
                cv2.putText(img, str(trackers[j].id), (x1, y1-20), font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)    ###Edit
                trash_count(img, trackers[j])


            print(time.time()-start)
            cv2.imshow('frame', img)


            key = cv2.waitKey(frameRate)
            if key == 27:
                break
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
