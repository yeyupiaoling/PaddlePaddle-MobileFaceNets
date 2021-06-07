import argparse
import functools
import os
import shutil
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.metric import accuracy
from paddle.static import InputSpec
from visualdl import LogWriter

from utils.mobilefacenet import MobileFaceNet
from utils.resnet import resnet_face34
from utils.ArcMargin import ArcNet
from utils.reader import CustomDataset
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpu',              str,    '0,1',                    '训练使用的GPU序号')
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    85742,                    '分类的类别数量')
add_arg('learning_rate',    float,  1e-1,                     '初始学习率的大小')
add_arg('use_model',        str,    'mobilefacenet',          '所使用的模型，支持mobilefacenet，resnet_face34')
add_arg('train_root_path',  str,    'dataset/images',         '训练数据的根目录')
add_arg('test_list_path',   str,    'dataset/lfw_test.txt',   '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/mobilefacenet',   '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练，当为None则不使用预训练模型，使用恢复训练模型最好同时也改学习率')
add_arg('pretrained_model', str,    None,                     '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def test(model):
    model.eval()
    # 获取测试数据
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    accuracy, _ = test_performance(fe_dict, args.test_list_path)
    model.train()
    return accuracy


# 保存模型
def save_model(args, model, optimizer):
    if not os.path.exists(os.path.join(args.save_model, 'params')):
        os.makedirs(os.path.join(args.save_model, 'params'))
    if not os.path.exists(os.path.join(args.save_model, 'infer')):
        os.makedirs(os.path.join(args.save_model, 'infer'))
    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(args.save_model, 'params/model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(args.save_model, 'params/optimizer.pdopt'))
    # 保存预测模型
    paddle.jit.save(layer=model,
                    path=os.path.join(args.save_model, 'infer/model'),
                    input_spec=[InputSpec(shape=[None, 3, 112, 112], dtype='float32')])


def train(args):
    # 设置支持多卡训练
    dist.init_parallel_env()
    if dist.get_rank() == 0:
        shutil.rmtree('log', ignore_errors=True)
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 获取数据
    train_dataset = CustomDataset(args.train_root_path, is_train=True)
    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)

    # 获取模型，贴心的作者同时提供了resnet的模型，以满足不同情况的使用
    if args.use_model == 'resnet_face34':
        model = resnet_face34()
    else:
        model = MobileFaceNet()
    metric_fc = ArcNet(feature_dim=512, class_dim=args.num_classes)
    if dist.get_rank() == 0:
        paddle.summary(model, input_size=(None, 3, 112, 112))
    # 设置支持多卡训练
    model = paddle.DataParallel(model)
    metric_fc = paddle.DataParallel(metric_fc)

    # 学习率衰减
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=10, gamma=0.1, verbose=True)
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters() + metric_fc.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(5e-4))

    # 加载预训练模型
    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        model_state_dict = paddle.load(os.path.join(args.pretrained_model, 'model.pdparams'))
        # 特征层
        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if weight.shape != list(model_state_dict[name].shape):
                    print('{} not used, shape {} unmatched with {} in model.'.
                            format(name, list(model_state_dict[name].shape), weight.shape))
                    model_state_dict.pop(name, None)
            else:
                print('Lack weight: {}'.format(name))
        model.set_dict(model_state_dict)
        print('成功加载 model 参数')

    # 恢复训练
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.resume, 'optimizer.pdopt')))
        print('成功加载模型参数和优化方法参数')

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        accuracies = []
        for batch_id, (img, label) in enumerate(train_loader()):
            feature = model(img)
            output = metric_fc(feature, label)
            # 计算损失值
            los = loss(output, label)
            loss_sum.append(los.numpy()[0])
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=output, label=label)
            accuracies.append(acc.numpy()[0])
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[%s] Train epoch %d, batch_id: %d, loss: %f, accuracy: %f' % (
                    datetime.now(), epoch, batch_id, sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies)))
                writer.add_scalar('Train loss', los, train_step)
                train_step += 1
                loss_sum = []
        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            acc = test(model)
            print('[%s] Train epoch %d, accuracy: %f' % (datetime.now(), epoch, acc))
            writer.add_scalar('Test acc', acc, test_step)
            # 记录学习率
            writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
            test_step += 1
            save_model(args, model, optimizer)
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    dist.spawn(train, args=(args,), gpus=args.gpu)
