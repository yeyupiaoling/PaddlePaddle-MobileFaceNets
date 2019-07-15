# 训练程序的配置信息
class TRAIN:
    # 数据的batch大小
    BATCH_SIZE = 8
    # 图片的通道数
    IMAGE_CHANNEL = 3
    # 图片的宽
    IMAGE_WIDTH = 112
    # 图片的高
    IMAGE_HEIGHT = 112
    # 训练的轮数
    PASS_SUM = 70
    # 数据的类别
    CLASS_DIM = 10177
    # 检查点模型路径
    PERSISTABLES_MODEL_PATH = '../model/persistable_model'
    # 训练的图像列表文件
    TRAIN_LIST = '../data/train_list.txt'
    # 测试的图像列表文件
    TEST_LIST = '../data/test_list.txt'
    # 是否使用GPU
    USE_GPU = True
    # 保存VisualDL日志的保存路径
    LOG_DIR = '../log'
    # 保存预测模型的路径
    SAVE_INFER_MODEL_PATH = 'model/infer_model'
    # 保存检查点模型的路径
    SAVE_PERSISTABLE_MODEL_PATH = 'model/persistable_model'


def show_train_args():
    print('------------------------------------------------')
    print('batch size:            %d' % TRAIN.BATCH_SIZE)
    print('image shape:           (%d, %d, %d)' % (TRAIN.IMAGE_CHANNEL, TRAIN.IMAGE_WIDTH, TRAIN.IMAGE_HEIGHT))
    print('pass sum:              %d' % TRAIN.PASS_SUM)
    print('class sum:             %d' % TRAIN.CLASS_DIM)
    print('use GPU:               %s' % TRAIN.USE_GPU)
    print('visualDL log path:     %s' % TRAIN.LOG_DIR)
    print('------------------------------------------------')
