import sys
sys.path.append('..')
import numpy as np
import mnist
import tempcoder
import torch
from Layer import Layer
def load_model_from_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    input = lines[0].split(' ')
    input_itr= iter(input)
    if input[0]=='pulses_per_layer':
        pulses_per_layer=True
        next(input_itr)
    else:
        pulses_per_layer=False
    threshold = float(next(input_itr))
    n_layers = int(next(input_itr))
    layer_sizes = []
    for _ in range(n_layers):
        layer_sizes.append(int(next(input_itr)))
    print("Loaded layer sizes: "+str(layer_sizes))
    print("The first layer is input!")
    n_pulse = int(next(input_itr))
    pulses = []
    if pulses_per_layer:
        for _ in range(n_layers):
            pulses.append([float(next(input_itr)) for _ in range(n_pulse)])
    else:
        pulses.append([float(next(input_itr)) for _ in range(n_pulse)])
        for _ in range(n_layers-1):
            pulses.append(pulses[0])
    print("Loaded pulses:")
    for pulse in pulses:
        print(pulse)

    layers = []
    for l in range(n_layers-1):
        layer = Layer(layer_sizes[l], layer_sizes[l+1], threshold, n_pulse=n_pulse)
        for i in range(layer.out_feature):
            for j in range(layer.in_feature+n_pulse):
                layer.weight[i][j]=float(next(input_itr))
        layer.pulse = torch.tensor(pulses[l])
        layers.append(layer)
    return layers

def encode(input, input_range=(0,1)):
    output = 1.0-input
    output = torch.where(output==1.0, tempcoder.kNoSpike, output*(input_range[1]-input_range[0])+input_range[0])
    return output

def test(layers, data, labels):
    correct = 0
    for i in range(len(labels)):
        x = encode(data[i])
        for l in layers:
            x = l.forward(x)
        max_i = torch.argmin(x)
        if x[max_i] == tempcoder.kNoSpike:
            max_i=-1
        correct+=(max_i==labels[i])
        print("Case: %d Pred: %d, Truth %d - %f" %(i+1, max_i, labels[i], correct/(i+1)), end='\r')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = torch.tensor(x_train, dtype=torch.float64)/255.0
    x_test = torch.tensor(x_test, dtype=torch.float64)/255.0
    layers = load_model_from_file('tempcoding/networks/slow_network')
    test(layers, x_test, y_test)