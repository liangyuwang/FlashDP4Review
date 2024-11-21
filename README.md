# FlashDP: Memory-Efficient and High-Throughput DP-SGD Training for Large Language Models

In the pursuit of balancing the privacy of data with the computational demands of Large Language Models (LLMs), we propose Flash-DP. Engineered with the core intent to minimize GPU memory consumption and maximize throughput, Flash-DP significantly enhances the efficiency of DP operations, particularly in models with a high parameter count such as GPT.

## Core Features

* Memory Optimization: Reduces GPU memory footprint, achieves memory usage almost parity with Non-DP methods, enabling larger models for LLM DP training.
* Throughput Maximization: Increases the number of operations per second, even surpasses Non-DP method in large matrix size, ensuring faster computations without compromising on privacy.
* Seamless Integration: Easily integrates with existing codebases, providing a straightforward path to enhanced DP performance.
* Versatile Optimization: While focused on Linear layers, FlashDP is adept at optimizing any layer that utilizes (Batched) Matrix Multiplication.
* LLM Specialization: Performing better in Large Matrix Multiplication of LLM.

## Getting Started

### Requirements

* Python >= 3.11
* CUDA >= 12.1
* PyTorch >= 2.3.1
* Triton >= 2.3.1
* transformers >= 4.38.2

### Installation

Clone and set up the FlashDP environment with the following commands:

```shell
cd FlashDP4Review
conda env create -f env.yml # Optional: Execute only if you wish to create a new Conda environment.
conda activate flashdp      # Optional: Run if the above command is executed.
bash install.sh
```

### Usage

FlashDP enables the seamless integration of differential privacy within your existing PyTorch models. Here’s how you can substitute key computational modules with their DP counterparts, as supported by FlashDP:

```python
import torch
from flashdp import wrap_with_flashdp_layers
# Define your pytorch model
model = your_pytorch_model.cuda()    # you can use different GPU
# Wrap the pytorch model with our FlashDP
model = wrap_with_flashdp_layers(
    model, 
    target_modules=[torch.nn.Linear, ], # you can add more supported modules
    skip_layers=['lm_head'], # skip the 'lm_head' layer from wrapping with FlashDP
    C=1.0,                  # hyperparameter: Gradient Norm Bound
    noise_multiplier=1.0    # hyperparameter: Noise Multiplier
)
# Normal torch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Normal training for loop
...
```

You can apply differential privacy techniques directly within your existing PyTorch model architectures by substituting key computational modules with their DP counterparts. This allows for a tailored approach to enhancing privacy without disrupting your model’s core functionalities. See [supported modules](##Support-target-modules) for more details on which modules can be enhanced with FlashDP.

You can also easily use "Opacus" to compute hyperparameters, or compute hyperparameters by yourself.

```python
from opacus.accountants.utils import get_noise_multiplier
noise_multiplier = get_noise_multiplier(target_epsilon=...,
                                        target_delta=...,
                                        epochs=...,
                                        steps=...,
                                        sample_rate=...,
                                        )
```

## Example

Try our example to train GPT2 with FlashDP. It requires an Ampere Graphics Processor (Ampere GPU) in order to flash attention and at least 8 GB of GPU memory.

```shell
# set "use_dp" to False if you do not want to use DP
python example/gpt2/train.py --use_dp=True
```

## Support target modules

* torch.nn.Linear
* torch.nn.LayerNorm
* transformers.pytorch_utils.Conv1D
