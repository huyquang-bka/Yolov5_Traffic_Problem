import cv2


def remove_object_overlapping(objects_xyxy, threshold=0.8):
    new_objects_xyxy = []
    for i in range(len(objects_xyxy)):
        x1, y1, x2, y2 = objects_xyxy[i]
        for j in range(i, len(objects_xyxy)):
            x3, y3, x4, y4 = objects_xyxy[j]
            xc1 = (x1 + x2) / 2
            yc1 = (y1 + y2) / 2
            if not (x3 < xc1 < x4 and y3 < yc1 < y4):
                continue
            x1_max = max(x1, x3)
            x2_min = min(x2, x4)
            y1_max = max(y1, y3)
            y2_min = min(y2, y4)
            area_intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
            if area_intersection / (x2 - x1) / (y2 - y1) > threshold:
                break
        new_objects_xyxy.append([x1, y1, x2, y2])
    return new_objects_xyxy


def draw_bounding_box(image, objects_xyxy_dictionary, color=(0, 0, 255), thickness=2):
    for object_id in objects_xyxy_dictionary.keys():
        x1, y1, x2, y2, conf = objects_xyxy_dictionary[object_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # cv2.putText(image, str(object_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def visualize_speed(image, speed_dict, new_dict):
    for key in speed_dict.keys():
        if key in new_dict:
            x1, y1, x2, y2 = new_dict[key][:4]
            cv2.putText(image, str(round(speed_dict[key], 2)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image