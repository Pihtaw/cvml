# app/model/onnx_infer.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import math
import json
from typing import List, Tuple

def run_onnx_detector(sess: ort.InferenceSession, img_pil: Image.Image, input_name: str, img_size: int = 640):
    """
    Выполняет инференс детектора на одном PIL изображении.
    Возвращает raw output (list of numpy arrays) из sess.run.
    """
    im = img_pil.convert('RGB').resize((img_size, img_size))
    arr = np.array(im).astype(np.float32) / 255.0
    # yolov5 export expects NCHW
    inp = np.transpose(arr, (2,0,1))[None,:,:,:].astype(np.float32)
    outputs = sess.run(None, {input_name: inp})
    return outputs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_yolov5_output(output, img_size=640, conf_thres=0.25):
    """
    Декодирует выход ONNX от yolov5 export.
    Ожидается формат [1, N, 5+num_classes] или [N, 85].
    Возвращает списки: boxes (x1,y1,x2,y2 в пикселях), scores, class_ids
    """
    # поддержка разных форматов
    if isinstance(output, list):
        out = output[0]
    else:
        out = output
    preds = np.array(out)
    if preds.ndim == 3:
        preds = preds[0]
    xywh = preds[:, :4]
    obj = preds[:, 4:5]
    cls_probs = preds[:, 5:]
    # если cls_probs суммарно >1, вероятно уже softmaxed; используем argmax
    class_ids = np.argmax(cls_probs, axis=1)
    class_scores = cls_probs[np.arange(len(class_ids)), class_ids]
    scores = (obj[:,0] * class_scores).astype(np.float32)
    # фильтр по порогу
    keep = scores > conf_thres
    if keep.sum() == 0:
        return np.zeros((0,4)), np.array([]), np.array([], dtype=int)
    xywh = xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    # Если xywh в нормализованных координатах (0..1), умножим на img_size
    # Определим по величинам: если max(xywh) <= 1.01 — нормализовано
    if xywh.max() <= 1.01:
        xywh = xywh * img_size
    cx = xywh[:,0]; cy = xywh[:,1]; w = xywh[:,2]; h = xywh[:,3]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    boxes = np.stack([x1,y1,x2,y2], axis=1)
    return boxes, scores, class_ids

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45):
    """
    Простая NMS на numpy. boxes: [N,4] x1,y1,x2,y2
    Возвращает индексы оставшихся боксов.
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)

def detections_from_onnx_output(outputs, img_size=640, conf_thres=0.25, iou_thres=0.45):
    """
    Полный pipeline: decode -> NMS -> вернуть финальные боксы, scores, classes
    """
    boxes, scores, class_ids = decode_yolov5_output(outputs, img_size=img_size, conf_thres=conf_thres)
    if boxes.shape[0] == 0:
        return [], [], []
    keep = nms_numpy(boxes, scores, iou_threshold=iou_thres)
    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_classes = class_ids[keep]
    return final_boxes, final_scores, final_classes

def bboxs_to_grid(boxes: np.ndarray, image_size: Tuple[int,int], grid_n: int = 19):
    """
    Преобразует список bbox (x1,y1,x2,y2) в карту grid_n x grid_n.
    Возвращает grid (grid_n,grid_n) с метками 0/1/2 и список назначений.
    При конфликте выбирается bbox с наибольшим score (это нужно делать с scores).
    """
    W, H = image_size
    grid = np.zeros((grid_n, grid_n), dtype=int)
    assignments = []  # list of (gy,gx,cls,box)
    if boxes is None or len(boxes)==0:
        return grid, assignments
    for b in boxes:
        x1,y1,x2,y2 = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gx = int(min(grid_n-1, math.floor(cx / (W / grid_n))))
        gy = int(min(grid_n-1, math.floor(cy / (H / grid_n))))
        assignments.append((gy, gx, b))
    return grid, assignments
