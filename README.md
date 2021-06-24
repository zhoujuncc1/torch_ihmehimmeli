# Torch Ihmehimmeli
Torch Ihmehimmeli is a torch version of [ihmehimmeli](https://github.com/google/ihmehimmeli)

It only supports Alpha Synaptic Function, not DualExponential
## Install

Lambert by PyTorch operator is slow so I write a cuda extension. Install first by

    python setup.py build_ext --inplace

or use the script

    sh install.sh

## Usage

Run Mnist example with default parameter:

    python mnist_torch.py


Test with trained weight. The accuracy is currently 0.9361 and being updated as training is slow.

    python mnist_torch.py --epochs 0 --restore mnist_mlp.pt

## Development Test
test dir contains code for development testing. Need to enter the directory to run.

    cd test
    python test_to_cpp.py


## ToDo:
1. Improve time performance
2. Support Convolution - How to backpropagate Convolution Layer
3. Support DualExponential
