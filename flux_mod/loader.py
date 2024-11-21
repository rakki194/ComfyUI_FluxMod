import torch
import torch.nn as nn
from safetensors.torch import safe_open
from .model import FluxParams, FluxMod
from .layers import Approximator
import comfy
import comfy.model_patcher
from comfy import model_management
import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds




class ExternalFlux(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Flux
      
    def __init__(self,):
        self.unet_config = {}
        self.latent_format = self.latent_format()
        self.unet_config["disable_unet_model_creation"] = True
    

class ExternalFluxModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([kwargs.get("guidance", 3.5)]))
        return out
    

def load_selected_keys(filename, exclude_keywords=[]):
    """Loads all keys from a safetensors file except those containing specified keywords.

    Args:
        filename: Path to the safetensors file.
        exclude_keywords: List of keywords to exclude.

    Returns:
        A dictionary containing the loaded tensors.
    """

    tensors = {}
    with safe_open(filename, framework="pt") as f:
        for key in f.keys():
            if not any(keyword in key for keyword in exclude_keywords):
                tensors[key] = f.get_tensor(key)
    return tensors


def cast_layers(module, layer_type, dtype, exclude_keywords=[]):
    """Casts layers in a module to the specified dtype, excluding layers with names containing keywords.

    Args:
        module (nn.Module): The module containing layers to cast.
        layer_type (type): The type of layers to cast (e.g., nn.Linear).
        dtype (torch.dtype): The target data type for casting.
        exclude_keywords (list, optional): A list of keywords to exclude from casting based on layer names. Defaults to [].

    Returns:
        None
    """

    for child_name, child in module.named_children():
        if isinstance(child, layer_type) and not any(keyword in child_name for keyword in exclude_keywords):
            # Cast the layer only if it's the desired type and not excluded
            # omit bias
            child.weight.data = child.weight.data.to(dtype=dtype)
        else:
            # Recursively apply to child modules
            cast_layers(child, layer_type, dtype, exclude_keywords)


def load_flux_mod(model_path, timestep_guidance_path, linear_dtypes=torch.bfloat16):

    # just load safetensors here
    state_dict = load_selected_keys(model_path, ["mod", "time_in", "guidance_in", "vector_in"])
    
    timestep_state_dict = comfy.utils.load_torch_file(timestep_guidance_path)
      

    param_count = sum([x.numel() for x in state_dict.values()])
    unet_dtype = torch.bfloat16

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()


    params=FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    )

    model_conf = ExternalFlux()
    model = ExternalFluxModel(
        model_conf,
        model_type=comfy.model_base.ModelType.FLUX,
        device=model_management.get_torch_device()
    )

    
    model.diffusion_model = FluxMod(params=params)

    model.diffusion_model.load_state_dict(state_dict)
    model.diffusion_model.distilled_guidance_layer = Approximator(64, 3072, 5120, 4)
    model.diffusion_model.distilled_guidance_layer.load_state_dict(timestep_state_dict)
    model.diffusion_model.dtype = unet_dtype
    model.diffusion_model.eval()
    model.diffusion_model.to(unet_dtype)
    # we cast to fp8 for mixed matmul ops but omit the picky and sensitive layers
    cast_layers(model.diffusion_model, nn.Linear, dtype=linear_dtypes, exclude_keywords=["img_in", "final_layer", "scale"])
    model.diffusion_model

    model_patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device = load_device,
        offload_device = offload_device,
    )
    return model_patcher
