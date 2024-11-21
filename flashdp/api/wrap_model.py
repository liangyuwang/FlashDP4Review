import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from typing import List, Type

from ..layers.linear import DPLinear
from ..layers.layernorm import DPLayerNorm
from ..layers.transformers_conv1d import DPConv1D


dp_supported_modules = {
    nn.Linear: DPLinear,
    nn.LayerNorm: DPLayerNorm,
    Conv1D: DPConv1D,
}

def wrap_with_flashdp_layers(
        model, 
        target_modules: List[Type[nn.Module]] = None, 
        target_layers: List[str] = None,
        skip_layers: List[str] = None, 
        priority_to_target: bool = False,
        feedback: bool = True,
        C = 1.0, 
        noise_multiplier = 1.0, 
        clamp_value = 1.0, 
    ) -> nn.Module:
    """
    Wrap specified layers of a PyTorch model with differential privacy modules.
    
    Args:
        model (nn.Module): The model to be modified.
        target_modules (list of nn.Module): Modules to be replaced with their DP counterparts.
            Supported modules: [
                torch.nn.Linear, 
                torch.nn.LayerNorm, 
                transformers.pytorch_utils.Conv1D,
            ]
        target_layers (List[str]): Specific layers to target for wrapping. Takes precedence over target_modules if provided.
        skip_layers (list of str): Names of modules to skip during wrapping. Please use full names for nested modules.
        priority_to_target (bool): If True, target_layers takes precedence over skip_layers when conflicts occur. Default is False.
        C (float): Gradient norm bound for DP. Default is 1.0.
        noise_multiplier (float): Multiplier for the noise to be added post-clipping. Default is 1.0.
        clamp_value (float): Maximum allowable gradient value before clipping. Default is 1.0.
    
    Returns:
        nn.Module: The modified model with DP layers where applicable.
    
    Example to wrap a GPT2 model all supported modules with FlashDP excpet the lm_head layer:
        from flashdp import wrap_model
        from transformers import GPT2LMHeadModel
        from transformers.pytorch_utils import Conv1D
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        dp_model = wrap_model(
            model, 
            target_modules=[Conv1D, nn.Linear, nn.LayerNorm], 
            skip_layers=['lm_head'],
            C=1.0,
            noise_multiplier=1.0,
        )
    
    Clip function:
        The clip function is defined in `flashdp/core/clip_fn.py` and is used to clip the gradients.
        Equation: `clip_fn(x) = x * min(C / (x_norm + 1e-10), clamp_value)`
    
    Raises:
        ValueError: If any of the target modules are not supported or if `target_modules` is empty.
    """
    print("Wrapping the model with FlashDP...")
    if target_modules is None and target_layers is None:
        raise ValueError("Either target_modules or target_layers must be specified.")
    if target_modules is None or len(target_modules) == 0:
        raise ValueError("target_modules cannot be empty and must be list of torch.nn.Module.")
    supported_modules = dp_supported_modules.keys()
    if not all(m in supported_modules for m in target_modules):
        raise ValueError(f"target_modules must be among {supported_modules}, got {target_modules}")

    # Initialize target_layers and skip_layers as empty lists if None
    target_layers = target_layers if target_layers is not None else []
    skip_layers = skip_layers if skip_layers is not None else []

    # Process layer groups and apply priority rules
    selected_layers, skipped_layers = _process_layer_groups(model, target_layers, skip_layers, priority_to_target)

    # Select layers to wrap based on priority rules
    selected_layers = _select_layers(model, target_layers, skip_layers, priority_to_target)

    # Validate target_modules and skip_layers
    if target_modules:
        _validate_target_modules(model, target_modules)
    if skip_layers:
        _validate_skip_layers(model, skipped_layers)

    # Wrap the selected layers with FlashDP modules
    _wrap_selected_layers(model, selected_layers, target_modules, C, clamp_value, noise_multiplier)

    # Print feedback
    if feedback:
        _print_layer_feedback(selected_layers, "Wrapped")
        _print_layer_feedback(skip_layers, "Skipped")
    print("Model wrapped with FlashDP.")
    
    print(f"FlashDP requires compilation for the first few iterations, which may take a while. " \
            "Skipping the first few iterations if you want to compare the performance.")

    return model


def _get_init_args(module):
    """
    Extract initialization arguments from a module to properly initialize its DP counterpart.
    
    Args:
        module (nn.Module): Module from which to extract initialization parameters.
        
    Returns:
        dict: Initialization arguments needed for the DP module.
    """
    if isinstance(module, nn.Linear):
        return {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias
        }
    elif isinstance(module, nn.LayerNorm):
        return {
            "normalized_shape": module.normalized_shape,
            "eps": module.eps,
            "elementwise_affine": module.elementwise_affine
        }
    elif isinstance(module, Conv1D):
        return {
            "nf": module.nf,
            "nx": module.weight.shape[0]
        }
    else:
        raise NotImplementedError(f"Unsupported module type: {type(module)}")


def _process_layer_groups(model, target_layers, skip_layers, priority_to_target):
    """
    Automatically expands target_layers and skip_layers to include all submodules.
    Applies priority rules to resolve conflicts.
    """
    all_layers = _collect_all_layer_names(model)
    expanded_target_layers = _expand_submodules(target_layers, all_layers)
    expanded_skip_layers = _expand_submodules(skip_layers, all_layers)

    if priority_to_target:
        final_layers = set(expanded_target_layers) - set(expanded_skip_layers)
        final_skipped_layers = set(expanded_skip_layers) - set(expanded_target_layers)
    else:
        final_layers = set(expanded_target_layers) - set(expanded_target_layers).intersection(set(expanded_skip_layers))
        final_skipped_layers = set(expanded_skip_layers)

    return list(final_layers), list(final_skipped_layers)


def _expand_submodules(layers, all_layers):
    """
    For each layer in layers, include all its submodules from all_layers.
    """
    expanded_layers = []
    for layer in layers:
        expanded_layers.extend([l for l in all_layers if l.startswith(layer + '.') or l == layer])
    return expanded_layers


def _select_layers(model, target_layers, skip_layers, priority_to_target):
    """
    Apply priority rules and select layers for wrapping.
    """
    all_layers = set(_collect_all_layer_names(model))
    selected_layers = set(target_layers) if target_layers else all_layers
    skip_layers_set = set(skip_layers) if skip_layers else set()

    if priority_to_target:
        final_layers = selected_layers - skip_layers_set
    else:
        final_layers = selected_layers.difference(selected_layers.intersection(skip_layers_set))

    return final_layers


def _wrap_selected_layers(model, selected_layers, target_modules, C, clamp_value, noise_multiplier):
    """
    Wrap the selected layers with appropriate DP modules based on the specified criteria.
    
    Args:
        model (nn.Module): The original model to modify.
        selected_layers (set): Set of layer names to be wrapped.
        target_modules (list): List of module types to be replaced.
        C (float): Clipping parameter for differential privacy.
        clamp_value (float): Maximum allowable gradient value before clipping.
        noise_multiplier (float): Multiplier for the noise added post-clipping.
    """
    def _replace_module_recursive(model, path=''):
        for child_name, child in model.named_children():
            full_name = f"{path}.{child_name}" if path else child_name
            if isinstance(child, tuple(target_modules)) and full_name in selected_layers:
                dp_module_class = dp_supported_modules[type(child)]
                child_init_args = _get_init_args(child)
                new_module = dp_module_class(**child_init_args, C=C, clamp_value=clamp_value, noise_multiplier=noise_multiplier)
                child_device = next(child.parameters()).device
                new_module = new_module.to(child_device)
                new_module.load_state_dict(child.state_dict())
                new_module.train(child.training)
                setattr(model, child_name, new_module)
            elif full_name not in selected_layers or not isinstance(child, tuple(target_modules)):
                _replace_module_recursive(child, full_name)
    _replace_module_recursive(model)


def _validate_target_modules(model: nn.Module, target_modules: List[Type[nn.Module]]) -> None:
    """
    Validate the target_modules list to ensure that the specified module types are used in the model.
    """
    module_types_used = set()
    def collect_module_types(module: nn.Module):
        for child in module.children():
            module_types_used.add(type(child))
            collect_module_types(child)
    
    collect_module_types(model)
    for target_module in target_modules:
        if target_module not in module_types_used:
            print(f"Warning: Module type '{target_module.__name__}' specified in target_modules is not used in the model.")


def _validate_skip_layers(model: nn.Module, skip_layers: List[str]) -> None:
    """
    Validate the skip_layers list to ensure all names exist in the model's hierarchy.
    
    Args:
        model (nn.Module): The model to be validated.
        skip_layers (list of str): Names of modules to be validated.
    
    Raises:
        Warning: If a module name in skip_layers does not exist in the model.
    """
    all_module_names = set()
    def _collect_module_names(module: nn.Module, prefix=''):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            all_module_names.add(path)
            _collect_module_names(child, path)
    
    _collect_module_names(model)
    
    for skip_module in skip_layers:
        if skip_module not in all_module_names:
            print(f"Warning: '{skip_module}' specified in skip_layers does not exist in the model.")


def _print_layer_feedback(layers, action):
    """
    Print which layers have been wrapped or skipped.
    """
    if layers:
        print(f"Layers {action}: {', '.join(layers)}")
    else:
        print(f"No layers {action}.")


def _collect_all_layer_names(model):
    """
    Collect all layer names from the model.
    """
    layer_names = []
    def collect_names(module, prefix=''):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            layer_names.append(path)
            collect_names(child, path)
    collect_names(model)
    return layer_names

