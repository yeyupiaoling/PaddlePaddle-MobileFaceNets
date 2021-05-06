import random
from skimage import transform as trans
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from paddle.io import Dataset


def random_brightness(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(img).enhance(e)


# 随机修改图片的对比度
def random_contrast(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(img).enhance(e)


# 随机修改图片的颜色强度
def random_color(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Color(img).enhance(e)


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    src = np.expand_dims(src, axis=0)
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112):
    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def process(img_path, landmark, image_size=112):
    img = cv2.imread(img_path)
    img = norm_crop(img, landmark, image_size)
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) / 128.0
    return img


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, train_list_path):
        super(CustomDataset, self).__init__()
        with open(train_list_path, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        img_path, id, landmarks = self.lines[idx].replace('\n', '').split('\t')
        landmarks = landmarks.split(',')
        landmarks = [[float(landmarks[i]), float(landmarks[i + 1])] for i in range(0, len(landmarks), 2)]
        landmarks = np.array(landmarks, dtype='float32')
        img = process(img_path, landmarks)
        return img, int(id)

    def __len__(self):
        return len(self.lines)
