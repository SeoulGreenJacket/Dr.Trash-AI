import random
import string
import numpy as np
from collections import deque
#from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.utils.linear_assignment_ import linear_assignment
def id_gen(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

class PersonTracker(object):
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=30)
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

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
        print(matched_idx)
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
if __name__ == '__main__':
    trackers = []
    for i in range(5):
        bboxes = [frames[i]]
        track_boxes = [tracker.bbox for tracker in trackers]
        matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b for b in bboxes])
        ###
        print('matched', matched)
        print('unmatched_tracker', unmatched_trackers)
        print('unmatched_detections', unmatched_detections)

        ###
        for idx, jdx in matched:
            print(bboxes[idx])     ###
            trackers[idx].set_bbox(bboxes[jdx])

        for idx in unmatched_detections:
            try:
                trackers.pop(idx)
            except:
                pass

        for idx in unmatched_trackers:
            person = PersonTracker()
            print('bboxes[idx]',bboxes[idx]) #####
            person.set_bbox(bboxes[idx])
            trackers.append(person)

        for tracker in trackers:
            print(tracker.id)