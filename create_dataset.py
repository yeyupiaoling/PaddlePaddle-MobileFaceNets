import numpy as np
import cv2
from skimage import transform as trans
from tqdm import tqdm


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


def main(list_path='dataset/train_list.txt', image_size=112):
    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        img_path, id, landmarks = line.replace('\n', '').split('\t')
        img = cv2.imread(img_path)
        if img.shape[0] == image_size and img.shape[1] == image_size:
            continue
        landmarks = landmarks.split(',')
        landmarks = [[float(landmarks[i]), float(landmarks[i + 1])] for i in range(0, len(landmarks), 2)]
        landmarks = np.array(landmarks, dtype='float32')
        img = norm_crop(img, landmarks, image_size)
        cv2.imwrite(img_path, img)


if __name__ == '__main__':
    main()
