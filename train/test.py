import copy
import random

import cv2
import numpy
import numpy as np

coord5point = [[51.50082, 87.88371],
               [111.40406, 87.88371],
               [81.64284, 121.95222],
               [57.03381, 157.02135],
               [106.64083, 157.02135]]

face_landmarks = [325.014, 151.109,
                  400.579, 168.538,
                  365.502, 204.401,
                  315.705, 240.369,
                  369.122, 251.158]


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
    print(len(orgi_landmarks))
    for i in range(0, len(orgi_landmarks), 2):
        trans_landmarks.append(width - orgi_landmarks[i])
        trans_landmarks.append(orgi_landmarks[i + 1])
    return trans_landmarks


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    if random.choice([0, 1]) > 0:
        img_im = cv2.flip(img_im, 1)
        orgi_landmarks = transpose_landmarks(img_im, face_landmarks)

    pts1 = numpy.float64(numpy.matrix(
        [[float(orgi_landmarks[i]), float(orgi_landmarks[i + 1])] for i in range(0, len(orgi_landmarks), 2)]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst, M[:2]


def draw_landmark(img_im, land):
    img = copy.deepcopy(img_im)
    for i in range(0, len(land), 2):
        cv2.circle(img, (int(land[i]), int(land[i + 1])), 2, (0, 255, 0), -1)
    cv2.imshow('aaa', img)


def main():
    pic_path = '20181216222654763.png'
    img_im = cv2.imread(pic_path)
    draw_landmark(img_im, orgi_landmarks)
    dst, M = warp_im(img_im, orgi_landmarks, coord5point)
    cv2.imshow('affine', dst)


if __name__ == '__main__':
    # img = cv2.imread('20181216222654763.png')
    # img = warp_im(img, face_landmarks, coord5point)
    # img = img[59:170, 29:140, :]
    main()
    cv2.waitKey()
    pass
