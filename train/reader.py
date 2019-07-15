import random
from multiprocessing import cpu_count
import config as cfg
import cv2
import numpy
import paddle
import uuid

coord5point = [[51.50082, 87.88371],
               [111.40406, 87.88371],
               [81.64284, 121.95222],
               [57.03381, 157.02135],
               [106.64083, 157.02135]]


def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])


def transpose_landmarks(img, orgi_landmarks):
    _, width, _ = img.shape
    trans_landmarks = []
    for i in range(0, len(orgi_landmarks), 2):
        trans_landmarks.append(width - float(orgi_landmarks[i]))
        trans_landmarks.append(float(orgi_landmarks[i + 1]))
    return trans_landmarks


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    if random.choice([0, 1]) > 0:
        img_im = cv2.flip(img_im, 1)
        orgi_landmarks = transpose_landmarks(img_im, orgi_landmarks)
    pts1 = numpy.float64(numpy.matrix(
        [[float(orgi_landmarks[i]), float(orgi_landmarks[i + 1])] for i in range(0, len(orgi_landmarks), 2)]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst


# 训练图片的预处理
def train_mapper(sample):
    img_path, label, bbox, landmarks = sample
    img = cv2.imread(img_path)
    img = warp_im(img, landmarks, coord5point)
    img = img[59:170, 29:140, :]
    img = cv2.resize(img, (cfg.TRAIN.IMAGE_WIDTH, cfg.TRAIN.IMAGE_HEIGHT))
    img = numpy.array(img).astype(numpy.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)


# 获取训练的reader
def train_reader(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            numpy.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img_path = line.split()[0:1][0]
                label = line.split()[1:2][0]
                bbox = line.split()[2:6]
                landmarks = line.split()[6:16]
                yield img_path, label, bbox, landmarks

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 10240)


# 测试图片的预处理
def test_mapper(sample):
    img_path, label, bbox, landmarks = sample
    img = cv2.imread(img_path)
    img = warp_im(img, landmarks, coord5point)
    img = img[59:170, 29:140, :]
    img = cv2.resize(img, (cfg.TRAIN.IMAGE_WIDTH, cfg.TRAIN.IMAGE_HEIGHT))
    img = numpy.array(img).astype(numpy.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img, int(label)


# 测试的图片reader
def test_reader(test_list_path):
    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.split()[0:1][0]
                label = line.split()[1:2][0]
                bbox = line.split()[2:6]
                landmarks = line.split()[6:16]
                yield img_path, label, bbox, landmarks

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 10240)
