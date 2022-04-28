import os
import sys
from pathlib import Path

from imutils.video import fps

import opt
import time

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from HuyQuang.utils import *
from HuyQuang.ObjectCounting import *

from utils.general import xyxy2xywh


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

source = str(opt.source)
# Load model
device = select_device(opt.device)
model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

# Half
half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.model.half() if opt.half else model.model.float()


def detect(img0):
    im0 = img0.copy()
    img = letterbox(img0, opt.imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=opt.augment, visualize=opt.visualize)

    # NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, agnostic=opt.agnostic_nms,
                               max_det=opt.max_det)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            xyxys, confs, clss = [], [], []
            for *xyxy, conf, cls in reversed(det):
                # label = '%s %.2f' % (names[int(cls)], conf)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                xyxys.append([x1, y1, x2, y2])
                confs.append(conf)
                clss.append(cls)
            xywhs = xyxy2xywh(torch.Tensor(xyxys))
            confs = torch.Tensor(confs)
            clss = torch.tensor(clss)
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
            tracking_dict = {}
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    x1, y1, x2, y2 = output[0:4]
                    # x_center = (x1 + x2) // 2
                    # y_center = (y1 + y2) // 2
                    id = output[4]
                    tracking_dict[id] = [x1, y1, x2, y2, conf]
                    # cls = output[5]
                    # c = int(cls)  # integer class
                    # color = (0, 0, 255)
                    # cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(im0, f"{id} {cls}", (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    try:
        return im0, tracking_dict
    except:
        return img0, {}


if __name__ == '__main__':
    path = r"data.mp4"
    cap = cv2.VideoCapture(path)
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(r"output.mp4", fourcc, vid_fps, (W, H))

    old_tracking_dict = {}
    object_passing_count = 0
    H_limit = H // 2 - 105
    H_limit_2 = H // 2 - 150
    old_dict_speed = {}
    speed_dict_save = {}
    frame_count = 0
    with torch.no_grad():
        while True:
            frame_count += 1
            t = time.time()
            ret, img0 = cap.read()
            H, W = img0.shape[:2]
            img0, new_tracking_dict = detect(img0)
            if new_tracking_dict:
                count, list_remove_key = count_objects_up(new_tracking_dict, old_tracking_dict, H_limit_2)
                object_passing_count += count
                for key in list_remove_key:
                    del old_tracking_dict[key]
                old_tracking_dict = add_to_dict(old_tracking_dict, new_tracking_dict, H_limit_2)
                # list_key_in_zone = get_key_in_zone(old_tracking_dict, H_limit)
                old_dict_speed = speed_phase_1(old_dict_speed, new_tracking_dict, H_limit, frame_count)
                speed_dict, list_remove_key_speed = speed_phase_2(old_dict_speed, new_tracking_dict, H_limit_2,
                                                                  frame_count)
                for key in speed_dict.keys():
                    if key not in speed_dict_save.keys():
                        speed_dict_save[key] = speed_dict[key]
                if len(speed_dict_save) > 100:
                    for key in sorted(speed_dict_save.keys())[50]:
                        del speed_dict_save[key]
                real_speed_dict = calculate_speed(speed_dict_save, vid_fps, range_pixel=45, real_length=3.5)
                img0 = draw_bounding_box(img0, new_tracking_dict)
                img0 = visualize_speed(img0, real_speed_dict, new_tracking_dict)
                for key in list_remove_key_speed:
                    if key in old_dict_speed:
                        del old_dict_speed[key]
                for key in speed_dict.keys():
                    if key in old_dict_speed:
                        del old_dict_speed[key]

            cv2.rectangle(img0, (0, 0), (300, 170), (0, 0, 0), thickness=-1)
            cv2.putText(img0, f"Object Counting: {object_passing_count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(img0, f"FPS: {1.0 / (time.time() - t):.2f}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255),
                        2)
            cv2.putText(img0, f"Objects in Frame: {len(new_tracking_dict)}", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(img0, f"White line length: 3,5m", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.line(img0, (0, H_limit), (W, H_limit), (0, 255, 0), 2)
            cv2.line(img0, (0, H_limit_2), (W, H_limit_2), (0, 255, 255), 2)
            out.write(img0)
            cv2.imshow("Image", img0)
            key = cv2.waitKey(1)
            # print("FPS: ", 1 // (time.time() - t))
            if key == ord("q"):
                break
            elif key == 32:
                cv2.waitKey()
