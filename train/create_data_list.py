import os
import random


# 获取图片的对于人脸的id
def get_face_identity(path):
    with open(path, 'r') as f1:
        lines = f1.readlines()
    data = []
    for line in lines:
        image_path, id = line.split()
        data.append([image_path, int(id) - 1])
    return data


# 获取人脸的box坐标
def get_face_box(path):
    with open(path, 'r') as f2:
        lines = f2.readlines()
    del lines[0]
    del lines[0]
    data = []
    for line in lines:
        image_path, x_1, y_1, width, height = line.split()
        data.append([image_path, x_1, y_1, width, height])
    return data


# 获取人脸的关键点坐标
def get_face_landmarks(path):
    with open(path, 'r') as f3:
        lines = f3.readlines()
    del lines[0]
    del lines[0]
    data = []
    for line in lines:
        image_path, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = line.split()
        data.append([image_path, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
                     rightmouth_x, rightmouth_y])
    return data


# 合并数据
def combine_data(path):
    identity_data = get_face_identity(os.path.join(path, 'identity_CelebA.txt'))
    box_data = get_face_box(os.path.join(path, 'list_bbox_celeba.txt'))
    landmarks_data = get_face_landmarks(os.path.join(path, 'list_landmarks_celeba.txt'))
    with open(os.path.join(path, 'data_list.txt'), 'w') as f:
        for i in range(len(identity_data)):
            image_path1, id = identity_data[i]
            image_path2, x_1, y_1, width, height = box_data[i]
            image_path3, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = \
            landmarks_data[i]
            if image_path1 == image_path2 and image_path2 == image_path3 and os.path.exists(os.path.join(path, 'img_celeba', image_path1)):
                f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                os.path.join(path, 'img_celeba', image_path1), id, x_1, y_1, width, height, lefteye_x, lefteye_y,
                righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y))


# 拆分训练数据列表和测试列表
def create_list(data_path, list_path):
    with open(list_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_f = open(os.path.join(data_path, 'train_list.txt'), 'w')
    test_f = open(os.path.join(data_path, 'test_list.txt'), 'w')
    for i in range(len(lines)):
        if i % 50 == 0:
            test_f.write(lines[i])
        else:
            train_f.write(lines[i])


if __name__ == '__main__':
    data_path = '../data'
    # 合并数据列表
    combine_data(data_path)
    # 拆分训练数据列表和测试列表
    create_list(data_path, os.path.join(data_path, 'data_list.txt'))