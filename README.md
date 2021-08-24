# Torch Ihmehimmeli
Torch Ihmehimmeli is a torch version of [ihmehimmeli](https://github.com/google/ihmehimmeli)

**Author:** Jun Zhou
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

## License & Copyright

Copyright 2021 Jun Zhou, torch_ihmehimmeli is free software: you can redistribute it and/or modoify it under the terms of GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ShenjingCat is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details http://www.gnu.org/licenses/.
