import argparse
import functools
import os
from datetime import datetime

import paddle
from paddle.static import InputSpec

from utils.mobilefacenet import MobileFaceNet
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('save_model',       str,    'models/',              '模型保存的路径')
add_arg('resume',           str,    'models/epoch_50',      '模型所在的文件夹路径')
args = parser.parse_args()
print_arguments(args)


# 获取模型
model = MobileFaceNet()

paddle.summary(model, input_size=(None, 3, 112, 112))

model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
print('[%s] 成功加载模型参数和优化方法参数' % datetime.now())


# 保存预测模型
if not os.path.exists(os.path.join(args.save_model, 'infer')):
    os.makedirs(os.path.join(args.save_model, 'infer'))
paddle.jit.save(layer=model,
                path=os.path.join(args.save_model, 'infer/model'),
                input_spec=[InputSpec(shape=[None, 3, 112, 112], dtype=paddle.float32)])
print('[%s] 模型导出成功：%s' % (datetime.now(), os.path.join(args.save_model, 'infer/model')))
