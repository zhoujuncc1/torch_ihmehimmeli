# Torch Ihmehimmeli
Torch Ihmehimmeli is a torch version of [ihmehimmeli](https://github.com/google/ihmehimmeli)

It only supports Alpha Synaptic Function, not DualExponential
## Install

Lambert and AlphaActivation by PyTorch operator is slow so cuda extension is written. Install first by

    python setup.py build_ext --inplace

or use the script

    sh install.sh

## Usage

Run Mnist example with default parameter:

    python mnist_torch.py


Test with trained weight. The accuracy is currently 0.9672. Accuracy from C++ ihmehimmeli is 0.974

    python mnist_torch.py --epochs 0 --restore mnist_mlp.pt

## Development Test
test dir contains code for development testing. Need to enter the directory to run.

    cd test
    python test_to_cpp.py
