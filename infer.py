import cv2
import numpy as np
import paddle.fluid as fluid

import config as cfg
import reader

# 获取执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 从保存的模型文件中获取预测程序、输入数据的名称和分类器
[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname=cfg.TRAIN.SAVE_INFER_MODEL_PATH,
                                                                               executor=exe,
                                                                               model_filename='model.paddle',
                                                                               params_filename='params.paddle')


# 对图片进行预处理
def load_image(img_path, landmarks):
    img_im = cv2.imread(img_path)
    img = reader.warp_im(img_im, landmarks, reader.coord5point)
    img = img[59:170, 29:140, :]
    img = cv2.resize(img, (cfg.TRAIN.IMAGE_WIDTH, cfg.TRAIN.IMAGE_HEIGHT))
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    return img


def infer(image_data):
    probs = []
    for data in image_data:
        img_path, landmarks = data
        # 添加待预测的图片
        infer_data = load_image(img_path, landmarks)[np.newaxis,]

        # 执行预测
        results = exe.run(program=infer_program,
                          feed={feeded_var_names[0]: infer_data},
                          fetch_list=target_vars)
        probs.append(results[0])

    # 对角余弦值
    dist = np.dot(probs[0], probs[1]) / (np.linalg.norm(probs[0]) * np.linalg.norm(probs[1]))
    print("两个人脸的相似度为：%f" % dist)


if __name__ == '__main__':
    images = [['093567.jpg', [87, 140, 140, 126, 119, 172, 106, 202, 152, 191]],
              ['089100.jpg', [75, 105, 123, 105, 97, 141, 76, 164, 118, 165]]]
    infer(images)
