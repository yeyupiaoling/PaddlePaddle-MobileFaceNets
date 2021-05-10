import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from paddle.io import Dataset
from skimage import transform as trans


def random_brightness(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(img).enhance(e)


# 随机修改图片的对比度
def random_contrast(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(img).enhance(e)


# 随机修改图片的颜色强度
def random_color(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Color(img).enhance(e)


# 水平翻转
def flip(img, landmark):
    h, w, c = img.shape
    img = cv2.flip(img, 1)
    landmark_ = []
    for l in landmark:
        landmark_.append([w - l[0], l[1]])
    landmark_ = np.array(landmark_, dtype=np.float32)
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return img, landmark_


# 对齐
def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    src = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    tform.estimate(lmk, src)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def process(img, landmark, image_size=112, is_train=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    if not is_train and random.random() > 0.5:
        img, landmark = flip(img, landmark)
    img = norm_crop(img, landmark, image_size)
    if is_train:
        # 转成PIL进行预处理
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ops = [random_brightness, random_contrast, random_color]
        np.random.shuffle(ops)
        if random.random() > 0.5:
            img = ops[0](img)
        # 转回cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) / 128.0
    return img


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, train_list_path, is_train=True):
        super(CustomDataset, self).__init__()
        with open(train_list_path, 'r') as f:
            self.lines = f.readlines()
        self.is_train = is_train

    def __getitem__(self, idx):
        img_path, id, landmarks = self.lines[idx].replace('\n', '').split('\t')
        landmarks = landmarks.split(',')
        landmarks = [[float(landmarks[i]), float(landmarks[i + 1])] for i in range(0, len(landmarks), 2)]
        landmarks = np.array(landmarks, dtype='float32')
        img = process(img_path, landmarks, is_train=self.is_train)
        img = np.array(img, dtype='float32')
        return img, np.array(int(id), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
