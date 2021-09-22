import argparse
import functools
import os
from datetime import datetime

import paddle
from paddle.static import InputSpec

from utils.reader import CustomDataset
from utils.rec_mv1_enhance import MobileFaceNet
from utils.resnet import resnet_face34
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'mobilefacenet',          '所使用的模型，支持 mobilefacenet，resnet_face34')
add_arg('train_root_path',  str,    'dataset/images',         '训练数据的根目录')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    'models/mobilefacenet/params/epoch_50',  '恢复训练，当为None则不使用恢复模型')
args = parser.parse_args()
print_arguments(args)

# 获取数据
train_dataset = CustomDataset(args.train_root_path, is_train=True)

# 获取模型，贴心的作者同时提供了resnet的模型，以满足不同情况的使用
if args.use_model == 'resnet_face34':
    model = resnet_face34()
else:
    model = MobileFaceNet()


paddle.summary(model, input_size=(None, 3, 112, 112))

model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
print('[%s] 成功加载模型参数和优化方法参数' % datetime.now())


# 保存预测模型
if not os.path.exists(os.path.join(args.save_model, args.use_model, 'infer')):
    os.makedirs(os.path.join(args.save_model, args.use_model, 'infer'))
paddle.jit.save(layer=model,
                path=os.path.join(args.save_model, args.use_model, 'infer/model'),
                input_spec=[InputSpec(shape=[None, 3, 112, 112], dtype=paddle.float32)])
print('[%s] 模型导出成功：%s' % (datetime.now(), os.path.join(args.save_model, args.use_model, 'infer/model')))
