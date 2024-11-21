import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import List
from tqdm.auto import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from flashdp import wrap_with_flashdp_layers

from model import GPT, GPTConfig

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__=="__main__":
    # Hyperparameters
    seed = 42
    use_dp = True   # Set to False if you do not want to use DP
    sequence_length = 1024  # Must be less than or equal to block_size
    epochs = 5
    batch_size = 4
    lr = 1e-4
    device = torch.device("cuda:0")
    use_amp = False # Currently, FlashDP will be slower when using AMP, we are working on it.

    exec(open('example/gpt2/configurator.py').read()) # overrides from command line or config file
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list, dict, type(None)))]
    config = {k: globals()[k] for k in config_keys} # will be useful for logging

    set_seed(seed)
    if use_amp:
        torch.set_float32_matmul_precision('high')

    # Initialize the GPT configuration
    config = GPTConfig()
    config.configure('gpt2-small')

    # Generate some data to CPU memory
    num_samples = 1024
    data = torch.randint(0, config.vocab_size, (num_samples, sequence_length+1))
    train_dataset = torch.utils.data.TensorDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)   # you can wrap DataLoader with different distribution

    # Create GPT model
    model = GPT(config).to(device)

    # Wrap the model with FlashDP 
    if use_dp:
        model = wrap_with_flashdp_layers(
            model, 
            target_modules=[torch.nn.Linear, torch.nn.LayerNorm], # you can add more supported modules
            skip_layers=['lm_head'], # skip the lm_head layer from wrapping with FlashDP
            C=1.0,                  # hyperparameter: Gradient Norm Bound
            noise_multiplier=1.0    # hyperparameter: Noise Multiplier
        )
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        global_loss_value = 0
        for idx, batch in enumerate(tqdm(train_loader)):
            input, label = batch[0][:,:-1].to(device), batch[0][:,1:].to(device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss_value = model.forward(input, label)
            else:
                _, loss_value = model.forward(input, label)
            global_loss_value += loss_value.item()
            
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Loss: {global_loss_value / len(train_loader)}")