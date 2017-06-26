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


def create_parser():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--log-level', default='INFO', help='sets the log level [default: INFO]')
    parser.add_argument('--ip', default='127.0.0.1', help='upstream ip that can recieve bboxes')
    parser.add_argument('--port', default=1337, type=int, help='remote port to connect to')
    return parser


def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname


def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)


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


def init_logging(args, kv):
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logger = logging.getLogger()

    handler = logging.FileHandler("log")
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(args.log_level.upper())


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
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    eval_metrics = ['accuracy']

    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, 50)]

    # take the first image of the validation dataset as example image for VisualBackProp
    first_batch = next(val_data)
    val_data.hard_reset()

    # build plotter and add it to all batch end callbacks
    plotter = VisualBackpropPlotter(upstream_ip=args.ip, upstream_port=args.port)
    batch_end_callbacks.append(
        plotter.get_callback(group, first_batch.data[0][0].asnumpy(), first_batch.label[0][0].asnumpy(), context, model)
    )

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
