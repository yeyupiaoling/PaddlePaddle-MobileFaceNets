import os
import pickle

import cv2
import numpy as np
from paddle import fluid


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def save_model(state_dicts, file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    def convert(state_dict):
        model_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, (fluid.framework.Variable, fluid.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(v, (fluid.framework.Variable, fluid.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)


def load_test_data(image_path, size=256):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    h, w, c = img.shape
    if img.shape[2] == 4:
        white = np.ones((h, w, 3), np.uint8) * 255
        img_rgb = img[:, :, :3].copy()
        mask = img[:, :, 3].copy()
        mask = (mask / 255).astype(np.uint8)
        img = (img_rgb * mask[:, :, np.newaxis]).astype(np.uint8) + white * (1 - mask[:, :, np.newaxis])

    img = cv2.resize(img, (size, size), cv2.INTER_AREA)
    img = RGB2BGR(img)

    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img


def preprocessing(x):
    x = x / 127.5 - 1
    return x


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h * j:h * (j + 1), w * i:w * (i + 1), :] = image

    return img


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


class BBox:
    # 人脸的box
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        """将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        """
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        """将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        """
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        """对所有关键点进行reproject操作"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        """对所有关键点进行project操作"""
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p


# 预处理数据，转化图像尺度并对像素归一
def processed_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    # 把图片转换成numpy值
    image = np.array(img_resized).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 归一化
    image = (image - 127.5) / 128
    return image


def convert_to_square(box):
    """将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻正方形最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def pad(bboxes, w, h):
    """将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    """
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    """

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def py_nms(dets, thresh, mode="Union"):
    """
    贪婪策略选择人脸框
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def generate_bbox(cls_map, reg, scale, threshold):
    """
     得到对应原图的box坐标，分类分数，box偏移量
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T
