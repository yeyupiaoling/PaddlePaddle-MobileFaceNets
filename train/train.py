import os
import shutil
import time

import paddle
import paddle.fluid as fluid
from visualdl import LogWriter

import config as cfg
import mobile_face_nets
import reader
from arc_margin_loss import ArcMarginLoss

# 打印配置参数
cfg.show_train_args()
# 创建保存VisualDL日志图的名称
log_writer = LogWriter(cfg.TRAIN.LOG_DIR, sync_cycle=10)
# 创建保存训练日志的VisualDL的工具
with log_writer.mode("train") as writer:
    train_cost_writer = writer.scalar('cost')
    train_accuracy_writer = writer.scalar('accuracy')
    histogram = writer.histogram("histogram", num_buckets=50)
# 创建保存测试日志的VisualDL的工具
with log_writer.mode("test") as writer:
    test_cost_writer = writer.scalar("cost")
    test_accuracy_writer = writer.scalar("accuracy")

# 定义输入层
image = fluid.layers.data(name='image', shape=[cfg.TRAIN.IMAGE_CHANNEL, cfg.TRAIN.IMAGE_WIDTH, cfg.TRAIN.IMAGE_HEIGHT],
                          dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取网络分类器
feature, net = mobile_face_nets.net(input=image, class_dim=cfg.TRAIN.CLASS_DIM)

# 获取损失函数和准确率函数
arc_loss = ArcMarginLoss(class_dim=cfg.TRAIN.CLASS_DIM)
cost, _ = arc_loss.loss(input=net, label=label)
# cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=net, label=label)

# 克隆测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
bd = [4, 7, 9, 11]
lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
optimizer = fluid.optimizer.AdamOptimizer(
    learning_rate=fluid.layers.piecewise_decay(boundaries=bd, values=lr),
    epsilon=0.1)
optimizer.minimize(avg_cost)

# 获取执行器并进行初始化
place = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 获取预测和训练数据
train_reader = paddle.batch(reader=reader.train_reader(cfg.TRAIN.TRAIN_LIST), batch_size=cfg.TRAIN.BATCH_SIZE)
test_reader = paddle.batch(reader=reader.train_reader(cfg.TRAIN.TEST_LIST), batch_size=cfg.TRAIN.BATCH_SIZE)

# 获取输入数据维度
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

# 日志的位置
train_step = 0
test_step = 0
param_name = fluid.default_startup_program().global_block().all_parameters()[0].name
last_test_acc = 0

# 开始训练
for pass_id in range(cfg.TRAIN.PASS_SUM):
    # 训练
    for batch_id, data in enumerate(train_reader()):
        start = time.time()
        train_cost, train_acc, param = exe.run(program=fluid.default_main_program(),
                                               feed=feeder.feed(data),
                                               fetch_list=[avg_cost, accuracy, param_name],
                                               use_program_cache=True)

        end = time.time()
        # 写入训练日志数据到VisualDL日志文件中
        train_step += 1
        train_cost_writer.add_record(train_step, train_cost[0])
        train_accuracy_writer.add_record(train_step, train_acc[0])
        histogram.add_record(train_step, param.flatten())

        if batch_id % 100 == 0:
            print("Pass:%d, Batch:%d, Cost:%f, Accuracy:%f, time:%f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0], end - start))

    # 测试
    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, accuracy])

        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])
    test_cost = sum(test_costs) / len(test_costs)
    test_acc = sum(test_accs) / len(test_accs)
    print("Test:%d, Cost:%f, Accuracy:%f" % (pass_id, test_cost, test_acc))

    # 写入测试日志数据到VisualDL日志文件中
    test_step += 1
    test_cost_writer.add_record(test_step, test_cost)
    test_accuracy_writer.add_record(test_step, test_acc)

    # 如果测试的准确率比上一个的测试准确率要好，那就保存模型
    if test_acc >= last_test_acc:
        # 保存预测模型
        shutil.rmtree(cfg.TRAIN.SAVE_INFER_MODEL_PATH, ignore_errors=True)
        os.makedirs(cfg.TRAIN.SAVE_INFER_MODEL_PATH)
        fluid.io.save_inference_model(dirname=cfg.TRAIN.SAVE_INFER_MODEL_PATH,
                                      feeded_var_names=[image.name],
                                      target_vars=[net],
                                      executor=exe,
                                      model_filename='model.paddle',
                                      params_filename='params.paddle')
        # 把最大测试准确率赋值给上一个最后的准确率
        last_test_acc = test_acc
