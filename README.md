# 人脸识别 MobileFaceNet

## 获取数据
1. 官方提供的下载地址：https://pan.baidu.com/s/1zw0KA1iYW41Oo1xZRuHkKQ 密码:zu3w
2. 把`img_celeba.7z`解压到`dataset`目录下
3. 把`identity_CelebA.txt`文件复制到`dataset`目录下
4. 把`list_landmarks_celeba.txt`文件复制到`dataset`目录下


## 训练
1. 执行`python3 create_data_list.py` 创建训练的数据列表
2. 执行`python3 train.py` 执行训练


## 预测
1. 执行`python3 infer.py` 识别这里两张人脸的相似度。
