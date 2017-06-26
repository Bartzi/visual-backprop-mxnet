import json

import mxnet as mx


def string_to_tuple(string):
    return string.strip('(').strip(')').split(',')


def combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature):
    data_input_node = nodes[node['inputs'][0][0]]
    try:
        data_input_symbol = intermediate_symbols["{}_output".format(data_input_node['name'])]
    except ValueError:
        data_input_symbol = intermediate_symbols[data_input_node['name']]
    averaged_feature_map = mx.symbol.mean(data_input_symbol, axis=1, keepdims=True)
    feature_map = scaled_feature * averaged_feature_map
    return feature_map


def build_visual_backprop_symbol(start_symbol, input_name=None):
    computational_graph = json.loads(start_symbol.tojson())
    nodes = computational_graph['nodes']

    assert nodes[-1]['op'] == 'Activation', 'Visual Backprop needs an activation node as starting point!'

    intermediate_symbols = start_symbol.get_internals()

    feature_map = mx.symbol.mean(start_symbol, axis=1, keepdims=True)
    for node in reversed(nodes):
        if input_name is not None and node['name'] == input_name or node['name'] == 'data':
            break

        node_attrs = node.get('attr', None)
        if node['op'] == 'Convolution':
            kernel_height, kernel_width = map(int, string_to_tuple(node_attrs['kernel']))
            pad_height, pad_width = map(int, string_to_tuple(node_attrs.get('pad', '(0, 0)')))
            scaled_feature = mx.symbol.Deconvolution(
                data=feature_map,
                weight=mx.symbol.ones((1, 1, kernel_height, kernel_width)),
                kernel=(kernel_height, kernel_width),
                pad=(pad_height, pad_width),
                num_filter=1
            )
            feature_map = combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature)
        elif node['op'] == 'Pooling':
            kernel_height, kernel_width = map(int, string_to_tuple(node_attrs['kernel']))
            stride_height, stride_width = map(int, string_to_tuple(node_attrs['stride']))
            scaled_feature = mx.symbol.Deconvolution(
                data=feature_map,
                weight=mx.symbol.ones((1, 1, kernel_height, kernel_width)),
                kernel=(kernel_height, kernel_width),
                stride=(stride_height, stride_width),
                num_filter=1,
            )
            feature_map = combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature)
        else:
            continue

    # normalize feature map
    min_value = mx.symbol.min(feature_map)
    max_value = mx.symbol.max(feature_map)
    feature_map = mx.symbol.broadcast_sub(feature_map, min_value)
    feature_map = mx.symbol.broadcast_mul(feature_map, 1.0 / (max_value - min_value))
    return feature_map
