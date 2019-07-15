import time
import paddle
import paddle.fluid as fluid
import config as cfg
import mobile_face_nets
import reader

# 定义输入数据
image = fluid.layers.data(name='image', shape=[cfg.TRAIN.IMAGE_CHANNEL, cfg.TRAIN.IMAGE_WIDTH, cfg.TRAIN.IMAGE_HEIGHT],
                          dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取网络分类器
_, net = mobile_face_nets.net(input=image, class_dim=cfg.TRAIN.CLASS_DIM)

# 定义损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=net, label=label)

# 获取评估的测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 创建执行器并进行初始化
place = fluid.CUDAPlace(0) if cfg.TRAIN.USE_GPU else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 加载检查点模型
fluid.io.load_persistables(executor=exe, dirname=cfg.TRAIN.SAVE_PERSISTABLE_MODEL_PATH)

# 获取测试数据
test_reader = paddle.batch(reader=reader.test_reader(cfg.TRAIN.TEST_LIST), batch_size=cfg.TRAIN.BATCH_SIZE)

# 定义输入数据的维度
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

# 开始预测测试数据
test_costs, test_accs = [], []
start = time.time()
for batch_id, data in enumerate(test_reader()):
    test_cost, test_acc = exe.run(program=test_program,
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost, accuracy])
    # 把没一个batch的预测结果记录下来
    test_costs.append(test_cost[0])
    test_accs.append(test_acc[0])

end = time.time()
# 求预测结果的平均值
test_cost = sum(test_costs) / len(test_costs)
test_acc = sum(test_accs) / len(test_accs)
test_time = end - start

print('Evaluate result: Cost:%f, Accuracy:%f, Eval time:%0.2fs' % (test_cost, test_acc, test_time))
