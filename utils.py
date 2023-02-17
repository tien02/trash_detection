import math
import cv2 as cv
import numpy as np
import torchvision

def collate_fn(batch):
    return tuple(zip(*batch))

def apply_nms(pred:dict, iou_threshold:float=0.3):
    # Apply Non Max Suppression
    '''
    Input: 
        pred: dict{
            'boxes' (Tensor),
            'labels' (Tensor),
            'scores' (Tensor)
        }: Model Prediction

        iou_threshold (float): NMS IOU threshold, discard box which IOU < iou_threshold
    Output:
        new_pred: dict{
            'boxes' (Tensor),
            'labels' (Tensor),
            'scores' (Tensor)
        }: Predictio after apply NMS
    '''
    
    box, _, score = pred.values()
    keep_idx = torchvision.ops.nms(boxes=box, scores=score, 
                                iou_threshold=iou_threshold)
    new_pred = {k: v[keep_idx] for k, v in pred.items()}
    return new_pred

def batch_nms(preds:list, iou_threhold:float=0.3):
    batch = []
    for pred in preds:
        pred_after_nms = apply_nms(pred, iou_threhold)
        batch.append(pred_after_nms)
    return batch

def draw_bounding_box(img, labels, boxes):
    # Draw Bounding Box
    '''
    Input:
        img: image (H, W, C)
        labels: label, can be list of label or string
        boxes: bounding box, can be list of bouding box or single list
    Output:
        Image with bouding box drawn
    '''
    
    img = img.copy()
    
    if isinstance(boxes, list):
        boxes = np.array(boxes)

    if isinstance(labels, list):
        labels = np.array(labels)

    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    
    if isinstance(labels, str) or isinstance(labels, int):
        labels = np.array([labels])

    if labels.ndim > 2:
        labels = labels.reshape((-1, 1))
    if boxes.ndim > 2:
        boxes = boxes.reshape((-1, 4))

    for box, label in zip(boxes, labels):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(x1 + box[2])
        y2 = int(y1 + box[3])

        img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0),3)
        img = cv.putText(img, str(label), (x1 - 4, y1 - 4), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

    return img

def xywh2xyxy(box: list):
    # Convert bouding box from xywh format to xyxy
    box = list(box)
    assert len(box) == 4, "Number of coordinate should be 4"

    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    box[2] = x2
    box[3] = y2
    
    return [math.ceil(i) for i in box]

def xywh2cxcywh(box: list):
    # Convert bouding box from xywh format to cxcywh
    box = list(box)
    assert len(box) == 4, "Number of coordinate should be 4"

    box[0] = box[0] + (box[2] / 2)
    box[1] = box[1] + (box[3] / 2)

    return [math.ceil(i) for i in box]