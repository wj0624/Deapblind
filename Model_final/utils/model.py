"""Some utils related to Keras models.

SPDX-License-Identifier: MIT
Copyright © 2017 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/ssd_detectors
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import os


def get_layers(model):
    """Collects all layers in a model and the models contained in the model."""
    layers = []
    def get(m):
        for l in m.layers:
            if l.__class__.__name__ in ['Model', 'Functional']:
                get(l)
            else:
                if l not in layers:
                    layers.append(l)
    get(model)
    return layers


def load_weights(model, filepath, layer_names=None):
    """Loads layer weights from a HDF5 save file.

    # Arguments
        model: Keras model
        filepath: Path to HDF5 file
        layer_names: List of strings, names of the layers for which the 
            weights should be loaded. List of tuples 
            (name_in_file, name_in_model), if the names in the file differ 
            from those in model.
    """
    filepath = os.path.expanduser(filepath)
    f = h5py.File(filepath, 'r')

    if layer_names == None:
        layer_names = f.attrs['layer_names']

    for name in layer_names:
        if type(name) in [tuple, list]:
            name_model = name[1]
            name_file = name[0]
        else:
            name_model = str(name, 'utf-8')
            name_file = name
        g = f[name_file]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            layer = model.get_layer(name_model)
            #assert layer is not None
        except ValueError:
            print('layer missing %s' % (name_model))
            print('    file  %s' % ([w.shape for w in weights]))
            continue
        try:
            #print('load %s' % (name_model))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (name_model))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    f.close()

def freeze_layers(model, trainable_conv_layers=0, trainable_bn_layers=0):
    """Set layers to none trainable.

    # Argumentes
        model: Keras model
        trainable_conv_layers: Number ob trainable convolution layers at 
            the end of the architecture.
        trainable_bn_layers: Number ob trainable batchnorm layers at the 
            end of the architecture.
    """
    layers = [l for l in model.layers if l.__class__.__name__ in ['Dense', 'Conv1D', 'Conv2D', 'Conv3D']]
    for i, l in enumerate(layers[::-1]):
        l.trainable = i < trainable_conv_layers

    layers = [l for l in model.layers if l.__class__.__name__ in ['BatchNormalization']]
    for i, l in enumerate(layers[::-1]):
        l.trainable = i < trainable_bn_layers


def calc_memory_usage(model, batch_size=1):
    """Compute the memory usage of a keras modell.
    
    # Arguments
        model: Keras model.
        batch_size: Batch size used for training.
    
    source: https://stackoverflow.com/a/46216013/445710
    """

    shapes_mem_count = 0
    #shapes_mem_count += np.sum([np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes]) for l in layers])
    counts_outputs = []
    for l in model.layers:
        shapes = []
        for n in l._inbound_nodes:
            if type(n.output_shapes) == list:
                shapes.extend(n.output_shapes)
            else:
                shapes.append(n.output_shapes)
        counts_outputs.append(np.sum([np.prod(s[1:]) for s in shapes]))
    shapes_mem_count += np.sum(counts_outputs)
    
    trainable_count = np.sum([np.prod(p.shape) for p in model.trainable_weights])
    non_trainable_count = np.sum([np.prod(p.shape) for p in model.non_trainable_weights])
    
    # each shape unit occupies 4 bytes in memory
    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    
    for s in ['Byte', 'KB', 'MB', 'GB', 'TB']:
        if total_memory > 1024:
            total_memory /= 1024
        else:
            break
    print('model memory usage %8.2f %s' % (total_memory, s))


def count_parameters(model):
    trainable_count = int(np.sum([np.prod(p.shape) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([np.prod(p.shape) for p in model.non_trainable_weights]))
    
    print('trainable     {:>16,d}'.format(trainable_count))
    print('non-trainable {:>16,d}'.format(non_trainable_count))
    
    return trainable_count + non_trainable_count


def plot_parameter_statistic(model, layer_types=['Dense', 'Conv2D'], trainable=True, non_trainable=True, outputs=False, channels=False):
    layer_types = [l.__name__ if type(l) == type else l for l in layer_types]
    layers = get_layers(model)
    layers = [l for l in layers if l.__class__.__name__ in layer_types]
    names = [l.name for l in layers]
    y = range(len(names))
    
    plt.figure(figsize=[12,max(len(y)//4,1)])
    
    offset = np.zeros(len(layers), dtype=int)
    legend = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if trainable:
        counts_trainable = [np.sum([np.prod(p.shape) for p in l.trainable_weights]) for l in layers]
        plt.barh(y, counts_trainable, align='center', color=colors[0])
        offset += np.array(counts_trainable, dtype=int)
        legend.append('trainable')
    if non_trainable:
        counts_non_trainable = [np.sum([np.prod(p.shape) for p in l.non_trainable_weights]) for l in layers]
        plt.barh(y, counts_non_trainable, align='center', color=colors[1],  left=offset)
        offset += np.array(counts_non_trainable, dtype=int)
        legend.append('non-trainable')
    if outputs:
        #counts_outputs = [np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes]) for l in layers]
        counts_outputs = []
        for l in layers:
            shapes = []
            for n in l._inbound_nodes:
                if type(n.output_shapes) == list:
                    shapes.extend(n.output_shapes)
                else:
                    shapes.append(n.output_shapes)
            counts_outputs.append(np.sum([np.prod(s[1:]) for s in shapes]))
        plt.barh(y, counts_outputs, align='center', color=colors[2], left=offset)
        offset += np.array(counts_outputs, dtype=int)
        legend.append('outputs')
    if channels:
        counts_channels = [l.output_shape[-1] for l in layers]
        plt.barh(y, counts_channels, align='center', color=colors[3], left=offset)
        offset += np.array(counts_channels, dtype=int)
        legend.append('channels')
    
    plt.yticks(y, names)
    plt.ylim(y[0]-1, y[-1]+1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.legend(legend)
    plt.show()


def calc_receptive_field(model, layer_name, verbose=False):
    """Calculate the receptive field related to a certain layer.
    
    # Arguments
        model: Keras model.
        layer_name: Name of the layer.
    
    # Return
        rf: Receptive field (w, h).
        es: Effictive stides in the input image.
        offset: Center of the receptive field associated with the first unit (x, y).
    """
    # TODO...
    
    fstr = '%-20s %-16s %-10s %-10s %-10s %-16s %-10s %-16s'
    if verbose:
        print(fstr % ('name', 'type', 'kernel', 'stride', 'dilation', 'receptive field', 'offset', 'effective stride'))
    l = model.get_layer(layer_name)
    rf = np.ones(2)
    es = np.ones(2)
    offset = np.zeros(2)
    
    while True:
        layer_type = l.__class__.__name__
        k, s, d = (1,1), (1,1), (1,1)
        p = 'same'
        if layer_type in ['Conv2D']:
            k = l.kernel_size
            d = l.dilation_rate
            s = l.strides
            p = l.padding
        elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
            k = l.pool_size
            s = l.strides
            p = l.padding
        elif layer_type in ['ZeroPadding2D']:
            p = l.padding
        elif layer_type in ['InputLayer', 'Activation', 'BatchNormalization']:
            pass
        else:
            print('unknown layer type %s %s' % (l.name, layer_type))
        
        k = np.array(k)
        s = np.array(s)
        d = np.array(d)
        
        ek = k + (k-1)*(d-1) # effective kernel size
        rf = rf * s + (ek-s)
        es = es * s
        
        if p == 'valid':
            offset += ek/2
            print(ek/2, offset)
        if type(p) == tuple:
            offset -= [p[0][0], p[1][0]]
            print([p[0][0], p[1][0]], offset)
        
        rf = rf.astype(int)
        es = es.astype(int)
        #offset = offset.astype(int)
        if verbose:
            print(fstr % (l.name, l.__class__.__name__, k, s, d, rf, offset, es))
        
        if layer_type == 'InputLayer':
            break
        
        input_name = l.input.name.split('/')[0]
        input_name = input_name.split(':')[0]
        l = model.get_layer(input_name)
    
    return rf, es, offset
