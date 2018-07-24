import json

import mxnet as mx


def normalize_feature_map(a_feature_map):
    min_value = mx.symbol.min(a_feature_map)
    max_value = mx.symbol.max(a_feature_map)
    a_feature_map = mx.symbol.broadcast_sub(a_feature_map, min_value)
    a_feature_map = mx.symbol.broadcast_mul(a_feature_map, 1.0 / (max_value - min_value))
    return a_feature_map


def string_to_tuple(string):
    return string.strip('(').strip(')').split(',')


def combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature):
    data_input_node = nodes[node['inputs'][0][0]]
    try:
        data_input_symbol = intermediate_symbols["{}_output".format(data_input_node['name'])]
    except ValueError:
        try:
            data_input_symbol = intermediate_symbols[data_input_node['name']]
        except ValueError:
            data_input_symbol = intermediate_symbols["{}_output0".format(data_input_node['name'])] # MY looks like lrn has output0

    averaged_feature_map = mx.symbol.mean(data_input_symbol, axis=1, keepdims=True)
	
    # MY by bojarski comment from 16/11/17 on https://github.com/mbojarski/VisualBackProp/issues/1
    averaged_feature_map = normalize_feature_map(averaged_feature_map)

    fixed_scaled_feature = mx.symbol.Crop(scaled_feature, averaged_feature_map)

    feature_map = fixed_scaled_feature * averaged_feature_map
    return feature_map


def build_visual_backprop_symbol(start_symbol, input_name=None):
    computational_graph = json.loads(start_symbol.tojson())
    nodes = computational_graph['nodes']

    assert nodes[-1]['op'] == 'Activation', 'Visual Backprop needs an activation node as starting point!'

    intermediate_symbols = start_symbol.get_internals()

    feature_map = mx.symbol.mean(start_symbol, axis=1, keepdims=True)

    # MY by bojarski comment from 16/11/17 on https://github.com/mbojarski/VisualBackProp/issues/1
    feature_map = normalize_feature_map(feature_map)

    for node in reversed(nodes):
        if input_name is not None and node['name'] == input_name or node['name'] == 'data':
            break

        node_attrs = node.get('attrs', None) # MY was 'attr' until around mxnet 1.0
        if node['op'] == 'Convolution':
            kernel_height, kernel_width = map(int, string_to_tuple(node_attrs['kernel']))
            stride_height, stride_width = map(int, string_to_tuple(
                node_attrs.get('stride', '(1, 1)')))  # MY for stride in convolution
            pad_height, pad_width = map(int, string_to_tuple(node_attrs.get('pad', '(0, 0)')))
            scaled_feature = mx.symbol.Deconvolution(
                data=feature_map,
                weight=mx.symbol.ones((1, 1, kernel_height, kernel_width)),
                kernel=(kernel_height, kernel_width),
                pad=(pad_height, pad_width),
                stride=(stride_height, stride_width),  
                adj=(stride_height - 1, stride_width - 1),
                num_filter=1
            )
            feature_map = combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature)
        elif node['op'] == 'Pooling':
            kernel_height, kernel_width = map(int, string_to_tuple(node_attrs['kernel']))
            stride_height, stride_width = map(int, string_to_tuple(
                node_attrs.get('stride', '(1, 1)')))  # MY needs default- theoretically could be pooling without stride..
            scaled_feature = mx.symbol.Deconvolution(
                data=feature_map,
                weight=mx.symbol.ones((1, 1, kernel_height, kernel_width)),
                kernel=(kernel_height, kernel_width),
                stride=(stride_height, stride_width),
                adj=(stride_height - 1,
                     stride_width - 1),  # MY this is sometimes needed- will take care of it with Crop if needed in combine_feature_maps
                num_filter=1,
            )
            feature_map = combine_feature_maps(intermediate_symbols, node, nodes, scaled_feature)
        else:
            continue

    # normalize feature map
    feature_map = normalize_feature_map(feature_map)
    return feature_map
