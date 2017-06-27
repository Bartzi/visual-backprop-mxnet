import argparse
import errno
import gzip
import logging
import os
import struct

import mxnet as mx
import numpy as np
import requests


def create_parser():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--gpus', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='the batch size')
    parser.add_argument('--num-epochs', type=int, default=10, help='the number of training epochs')
    parser.add_argument('--kv-store', default='local', help='the kvstore type')
    parser.add_argument('--log-level', default='INFO', help='sets the log level [default: INFO]')
    parser.add_argument('--ip', default='127.0.0.1', help='upstream ip that can recieve bboxes [default: 127.0.0.1]')
    parser.add_argument('--port', default=1337, type=int, help='remote port to connect to [default: 1337]')
    return parser


def download_file(url, local_fname=None, force_write=False):
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