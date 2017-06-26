import socket

import base64

import json
import mxnet as mx
import numpy as np

from PIL import Image
from collections import namedtuple


Batch = namedtuple('Batch', ['data', 'label'])


class VisualBackpropPlotter:
    def __init__(self, upstream_ip='127.0.0.1', upstream_port=1337):
        socket.setdefaulttimeout(2)
        self.upstream_ip = upstream_ip
        self.upstream_port = upstream_port
        self.send_image = True

    def send_data(self, data):
        height = data.height
        width = data.width
        channels = len(data.getbands())
        data = np.asarray(data, dtype=np.uint8).tobytes()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.upstream_ip, self.upstream_port))
            except Exception as e:
                print(e)
                print("could not connect to display server, disabling image rendering")
                self.send_image = False
                return
            data = {
                'width': width,
                'height': height,
                'channels': channels,
                'image': base64.b64encode(data).decode('utf-8'),
            }
            sock.send(bytes(json.dumps(data), 'utf-8'))

    def get_callback(self, symbol, data, label, context, model):
        def plot_visual_backprop(execution_params):
            # build new module and copy current params
            data_iter = execution_params.locals['train_data']
            batch_size = 1

            input_data_shapes = [(description.name, (batch_size, ) + description.shape[1:]) for description in data_iter.provide_data]
            label_shapes = [(description.name, (batch_size, ) + description.shape[1:]) for description in data_iter.provide_label]

            executor = mx.module.Module(context=context, symbol=symbol)
            executor.bind(input_data_shapes, label_shapes, for_training=False, grad_req='null')

            arg_params, aux_params = model.get_params()
            executor.set_params(arg_params, aux_params)

            batch = Batch(data=[mx.nd.array(data[np.newaxis, ...])], label=[mx.nd.array(label)])

            # perform forward pass and get output, especially visualization output
            executor.forward(batch, is_train=False)

            visualization = executor.get_outputs()[1].asnumpy()
            visualization = np.tile(visualization[0], (3, 1, 1))
            visualization_image = Image.fromarray((visualization * 255).astype(np.uint8).transpose(1, 2, 0), "RGB")

            image = np.tile(data, (3, 1, 1))
            image = Image.fromarray((image * 255).astype(np.uint8).transpose(1, 2, 0), "RGB")

            # put visualization and original image in one large image and display it to the user
            dest_image = Image.new(
                "RGB",
                (image.width + visualization_image.width, max(image.height, visualization_image.height))
            )
            dest_image.paste(image, (0, 0))
            dest_image.paste(visualization_image, (image.width, 0))
            dest_image = dest_image.convert("RGBA")

            if self.send_image:
                self.send_data(dest_image)

        return plot_visual_backprop
