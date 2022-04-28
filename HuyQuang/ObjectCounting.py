import numpy as np
import cv2


def count_objects_up(new_dict, old_dict, H_limit):
    count = 0
    list_remove_key = []
    for key in new_dict:
        if key in old_dict:
            if old_dict[key][3] > H_limit > new_dict[key][3]:
                list_remove_key.append(key)
                count += 1
    return count, list_remove_key


def add_to_dict(old_dict, new_dict, H_limit):
    for key in new_dict:
        if H_limit < new_dict[key][3]:
            old_dict[key] = new_dict[key]
    return old_dict


def check_point_in_polygon(point, polygon):
    polygon = np.array(polygon)
    dist = cv2.pointPolygonTest(polygon, point, False)
    return True if dist >= 0 else False


def get_key_in_zone(dict_point, zone):
    list_key_in_zone = []
    for key in dict_point.keys():
        point = tuple(dict_point[key][:4:3])
        if check_point_in_polygon(point, zone):
            list_key_in_zone.append(key)
    return list_key_in_zone


def speed_dict_update(speed_dict, list_key_in_zone, frame_number):
    for key in list_key_in_zone:
        try:
            speed_dict[key]
        except:
            speed_dict[key] = []
    for key in list_key_in_zone:
        speed_dict[key].append(frame_number)
    return speed_dict


def speed_phase_1(old_speed_dict, new_dict, H_limit, frame_number, pixel_limit=10):
    for key in new_dict:
        if 0 < H_limit - new_dict[key][3] <= pixel_limit:
            try:
                old_speed_dict[key]
            except:
                old_speed_dict[key] = [frame_number, new_dict[key][3]]
    return old_speed_dict


def speed_phase_2(old_speed_dict, new_dict, H_limit_2, frame_number, pixel_limit=10):
    list_remove_key_speed = []
    speed_dict = {}
    for key in new_dict:
        if 0 <= H_limit_2 - new_dict[key][3] <= pixel_limit:
            try:
                speed_dict[key] = [frame_number - old_speed_dict[key][0], old_speed_dict[key][1] - new_dict[key][3]]
            except:
                pass
        elif H_limit_2 - new_dict[key][3] > pixel_limit:
            list_remove_key_speed.append(key)
    return speed_dict, list_remove_key_speed


def calculate_speed(speed_dict, fps, range_pixel=45, real_length=3):
    real_spped_dict = {}
    for key in speed_dict:
        frame_count, pixels = speed_dict[key]
        real_spped_dict[key] = (pixels / range_pixel) * real_length / (frame_count / fps) * 3.5
    return real_spped_dict
