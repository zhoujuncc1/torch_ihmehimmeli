# Torch Ihmehimmeli
Torch Ihmehimmeli is a torch version of [ihmehimmeli](https://github.com/google/ihmehimmeli)

It only supports Alpha Synaptic Function, not DualExponential
## Install and Use

Install cuda extension by

    sh install.sh

Run Example:

    python mnist_torch.py

## Test using trained weight

The accuracy is currently 0.9361 and being updated as training is slow.

    python mnist_torch.py --epochs 0 --restore mnist_mlp.pt

## Development Test
test dir contains code for development testing. Need to enter the directory to run.

    cd test
    python test_to_cpp.py


## ToDo:
1. Improve time performance
2. Support Convolution - How to backpropagate Convolution Layer
3. Support DualExponential
