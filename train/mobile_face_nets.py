import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input, filter_size, num_filters, stride, padding, num_groups=1, if_act=True, use_cudnn=True):
    conv = fluid.layers.conv2d(input=input,
                               num_filters=num_filters,
                               filter_size=filter_size,
                               stride=stride,
                               padding=padding,
                               groups=num_groups,
                               use_cudnn=use_cudnn,
                               param_attr=ParamAttr(),
                               bias_attr=False)
    bn = fluid.layers.batch_norm(input=conv,
                                 param_attr=ParamAttr(),
                                 bias_attr=ParamAttr())
    if if_act:
        return fluid.layers.prelu(bn, mode='channel')
    else:
        return bn


def shortcut(input, data_residual):
    return fluid.layers.elementwise_add(input, data_residual)


def inverted_residual_unit(input,
                           num_in_filter,
                           num_filters,
                           ifshortcut,
                           stride,
                           filter_size,
                           padding,
                           expansion_factor):
    num_expfilter = int(round(num_in_filter * expansion_factor))

    channel_expand = conv_bn_layer(input=input,
                                   num_filters=num_expfilter,
                                   filter_size=1,
                                   stride=1,
                                   padding=0,
                                   num_groups=1,
                                   if_act=True)

    bottleneck_conv = conv_bn_layer(input=channel_expand,
                                    num_filters=num_expfilter,
                                    filter_size=filter_size,
                                    stride=stride,
                                    padding=padding,
                                    num_groups=num_expfilter,
                                    if_act=True,
                                    use_cudnn=False)

    linear_out = conv_bn_layer(input=bottleneck_conv,
                               num_filters=num_filters,
                               filter_size=1,
                               stride=1,
                               padding=0,
                               num_groups=1,
                               if_act=False)
    if ifshortcut:
        out = shortcut(input=input, data_residual=linear_out)
        return out
    else:
        return linear_out


def invresi_blocks(input, in_c, t, c, n, s):
    first_block = inverted_residual_unit(input=input,
                                         num_in_filter=in_c,
                                         num_filters=c,
                                         ifshortcut=False,
                                         stride=s,
                                         filter_size=3,
                                         padding=1,
                                         expansion_factor=t)

    last_residual_block = first_block
    last_c = c

    for i in range(1, n):
        last_residual_block = inverted_residual_unit(input=last_residual_block,
                                                     num_in_filter=last_c,
                                                     num_filters=c,
                                                     ifshortcut=True,
                                                     stride=1,
                                                     filter_size=3,
                                                     padding=1,
                                                     expansion_factor=t)
    return last_residual_block


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride, padding, if_act):
    depthwise_conv = conv_bn_layer(input=input,
                                   filter_size=3,
                                   num_filters=num_filters1,
                                   stride=stride,
                                   padding=padding,
                                   num_groups=num_groups,
                                   use_cudnn=False,
                                   if_act=if_act)

    pointwise_conv = conv_bn_layer(input=depthwise_conv,
                                   filter_size=1,
                                   num_filters=num_filters2,
                                   stride=1,
                                   padding=0,
                                   if_act=if_act)
    return pointwise_conv


def net(input, class_dim):
    bottleneck_params_list = [
        (2, 64, 5, 2),
        (4, 128, 1, 2),
        (2, 128, 6, 1),
        (4, 128, 1, 2),
        (2, 128, 2, 1),
    ]

    # conv 3*3
    input = conv_bn_layer(input,
                          num_filters=64,
                          filter_size=3,
                          stride=2,
                          padding=1,
                          if_act=True)

    # detphwise conv 3*3
    input = depthwise_separable(input,
                                num_filters1=64,
                                num_filters2=64,
                                num_groups=64,
                                stride=1,
                                padding=1,
                                if_act=True)

    # bottleneck
    in_c = 32
    for layer_setting in bottleneck_params_list:
        t, c, n, s = layer_setting
        input = invresi_blocks(input=input,
                               in_c=in_c,
                               t=t,
                               c=c,
                               n=n,
                               s=s)
        in_c = c

    # conv 1*1
    input = conv_bn_layer(input=input,
                          num_filters=512,
                          filter_size=1,
                          stride=1,
                          padding=0,
                          if_act=True)

    # linear GDConv 7*7
    input = depthwise_separable(input,
                                num_filters1=512,
                                num_filters2=512,
                                num_groups=512,
                                stride=1,
                                padding=0,
                                if_act=False)
    # linear conv 1*1
    feature = conv_bn_layer(input=input,
                            num_filters=128,
                            filter_size=1,
                            stride=1,
                            padding=0,
                            if_act=False)

    net = fluid.layers.fc(input=feature,
                          size=class_dim,
                          act='softmax')
    return feature, net
