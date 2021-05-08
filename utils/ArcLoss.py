import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ArcSoftmax(nn.Layer):
    def __init__(self, feature_dim=512, class_dim=1000):
        super(ArcSoftmax, self).__init__()
        # 生成一个隔离带向量，训练这个向量和原来的特征向量分开，达到增加角度的目的
        self.W = paddle.to_tensor(paddle.randn((feature_dim, class_dim), dtype='float32'), stop_gradient=False)

    def forward(self, feature, m=1, s=10):
        # 对特征维度进行标准化
        x = F.normalize(feature, axis=1)
        w = F.normalize(self.W, axis=0)
        # 做L2范数化，将cosa变小，防止acosa梯度爆炸
        cosa = paddle.matmul(x, w) / s

        # 反三角函数得出的是弧度，而非角度
        a = paddle.acos(cosa)

        arcsoftmax = paddle.exp(s * paddle.cos(a + m)) / (paddle.sum(paddle.exp(s * cosa), axis=1, keepdim=True)
                                                          - paddle.exp(s * cosa) + paddle.exp(s * paddle.cos(a + m)))

        return arcsoftmax
