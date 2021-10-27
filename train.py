import argparse
import functools
import os
import shutil
import time
from datetime import datetime, timedelta

import paddle
from paddle.io import DataLoader
from paddle.metric import accuracy
from visualdl import LogWriter

from utils.arcmargin import ArcNet
from utils.mobilefacenet import MobileFaceNet
from utils.reader import CustomDataset
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_root_path',  str,    'dataset/train_data',     '训练数据的根目录')
add_arg('test_list_path',   str,    'dataset/lfw_test.txt',   '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练，当为None则不使用恢复模型')
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
def save_model(args, epoch, model, metric_fc, optimizer):
    model_params_path = os.path.join(args.save_model, 'epoch_%d' % epoch)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数
    paddle.save(model.state_dict(), os.path.join(model_params_path, 'model.pdparams'))
    paddle.save(metric_fc.state_dict(), os.path.join(model_params_path, 'metric_fc.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pdopt'))
    # 删除旧的模型
    old_model_path = os.path.join(args.save_model, 'epoch_%d' % (epoch - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)


def train(args):
    shutil.rmtree('log', ignore_errors=True)
    # 日志记录器
    writer = LogWriter(logdir='log')
    # 获取数据
    train_dataset = CustomDataset(args.train_root_path, is_train=True)
    # 设置支持多卡训练
    batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
    print("[%s] 总数据类别为：%d" % (datetime.now(), train_dataset.num_classes))

    # 获取模型
    model = MobileFaceNet(scale=1.0)
    metric_fc = ArcNet(feature_dim=1024, class_dim=train_dataset.num_classes)
    paddle.summary(model, input_size=(None, 3, 112, 112))

    # 初始化epoch数
    last_epoch = 0
    # 学习率衰减
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.learning_rate, step_size=1, gamma=0.8)
    # 设置优化方法
    optimizer = paddle.optimizer.Momentum(parameters=model.parameters() + metric_fc.parameters(),
                                          learning_rate=scheduler,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(1e-5))

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
        print('[%s] 成功加载 model 参数' % datetime.now())

    # 恢复训练
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
        metric_fc.set_state_dict(paddle.load(os.path.join(args.resume, 'metric_fc.pdparams')))
        optimizer_state = paddle.load(os.path.join(args.resume, 'optimizer.pdopt'))
        optimizer.set_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = optimizer_state['LR_Scheduler']['last_epoch']
        print('[%s] 成功加载模型参数和优化方法参数' % datetime.now())

    # 获取损失函数
    loss = paddle.nn.CrossEntropyLoss()
    train_step = 0
    test_step = 0
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        loss_sum = []
        accuracies = []
        start = time.time()
        for batch_id, (img, label) in enumerate(train_loader()):
            feature = model(img)
            output = metric_fc(feature, label)
            # 计算损失值
            los = loss(output, label)
            los.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 计算准确率
            label = paddle.reshape(label, shape=(-1, 1))
            acc = accuracy(input=paddle.nn.functional.softmax(output), label=label)
            accuracies.append(acc.numpy()[0])
            loss_sum.append(los.numpy()[0])
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f, lr: %f, eta: %s' % (
                    datetime.now(), epoch, batch_id, len(train_loader), sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies), scheduler.get_lr(), eta_str))
                writer.add_scalar('Train loss', los, train_step)
                train_step += 1
                loss_sum = []
            if batch_id % 20000 == 0 and batch_id != 0:
                save_model(args, epoch, model, metric_fc, optimizer)
            start = time.time()
        # 多卡训练只使用一个进程执行评估和保存模型
        print('='*70)
        acc = test(model)
        print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
        print('='*70)
        writer.add_scalar('Test acc', acc, test_step)
        # 记录学习率
        writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
        test_step += 1
        save_model(args, epoch, model, metric_fc, optimizer)
        scheduler.step()
    save_model(args, args.num_epoch, model, metric_fc, optimizer)


if __name__ == '__main__':
    print_arguments(args)
    train(args)
