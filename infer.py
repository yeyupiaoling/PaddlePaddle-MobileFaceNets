import argparse
import functools
import os

import cv2
import numpy as np
import paddle

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('image_path',               str,     'temp/test.jpg',                    '预测图片路径')
add_arg('face_db_path',             str,     'face_db',                          '人脸库路径')
add_arg('threshold',                float,   0.7,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'models/mobilefacenet/infer/model', 'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'models/mtcnn',                     'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)

        # 加载模型
        self.model = paddle.jit.load(mobilefacenet_model_path)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = {}
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            imgs = self.process(imgs)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            feature = self.infer(imgs[0])
            faces_db[name] = feature[0]
        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, img):
        assert len(img.shape) == 3 or len(img.shape) == 4
        if len(img.shape) == 3:
            img = img[np.newaxis, :]
        img = paddle.to_tensor(img, dtype='float32')
        # 执行预测
        feature = self.model(img)
        return feature.numpy()

    def recognition(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        imgs, boxes = self.mtcnn.infer_image(img)
        imgs = self.process(imgs)
        if imgs is None:
            return None, None
        imgs = np.array(imgs, dtype='float32')
        features = self.infer(imgs)
        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i]
            results_dict = {}
            for name in self.faces_db.keys():
                feature1 = self.faces_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            print(results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names

    # 画出人脸框和关键点
    @staticmethod
    def draw_face(image_path, boxes_c, names):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的置信度
                cv2.putText(img, name, (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("result", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path)
    boxes, names = predictor.recognition(args.image_path)
    print(boxes)
    print(names)
    predictor.draw_face(args.image_path, boxes, names)
