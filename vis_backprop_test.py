import argparse
import errno
import gzip
import logging
import os
import struct

import mxnet as mx
import numpy as np

from insights.iter_visualisation import VisualBackpropPlotter
from insights.visual_backprop import build_visual_backprop_symbol

from utils.train_utils import create_parser, download_file, get_mnist_iter, init_logging, read_data, to4d

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')

    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2, 2), stride=(2, 2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="relu")

    # create visual backprop anchor
    vis = build_visual_backprop_symbol(tanh2)

    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2, 2), stride=(2, 2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet, vis


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kv = mx.kvstore.create(args.kv_store)
    init_logging(args, kv)

    # create symbol and save visualization anchor
    net, vis = get_symbol()
    # group symbol for visualization pass
    group = mx.symbol.Group([net, vis])

    train_data, val_data = get_mnist_iter(args, kv)
    context = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    # create training module
    model = mx.mod.Module(
        context=mx.gpu(),
        symbol=net
    )
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, 50)]

    # take the first image of the validation dataset as example image for VisualBackProp
    first_batch = next(val_data)
    val_data.hard_reset()

    # build plotter and add it to all batch end callbacks
    plotter = VisualBackpropPlotter(upstream_ip=args.ip, upstream_port=args.port)
    batch_end_callbacks.append(
        plotter.get_callback(group, first_batch.data[0][0].asnumpy(), first_batch.label[0][0].asnumpy(), context, model)
    )

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    eval_metrics = ['accuracy']

    # start the training
    model.fit(
        train_data,
        begin_epoch=0,
        num_epoch=args.num_epochs,
        eval_data=val_data,
        eval_metric=eval_metrics,
        kvstore=kv,
        optimizer=mx.optimizer.Adam(),
        initializer=initializer,
        batch_end_callback=batch_end_callbacks,
        allow_missing=True,
    )
