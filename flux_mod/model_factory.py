import comfy.ops
import torch.nn as nn

try:
    from .model import FluxMod, FluxParams
    from .layers import Approximator
except ImportError:
    from ..model import (
        FluxMod,
        FluxParams,
    )
    from ..layers import Approximator


def get_flux_model_instance(
    hidden_size: int,
    num_heads: int,
    depth: int,
    depth_single_blocks: int,
    **other_kwargs,
):
    """
    Factory function to create a FluxMod instance.
    It takes individual parameters, creates a FluxParams object,
    and then instantiates FluxMod.
    """
    # Based on your docs/chroma_model_layout_for_svdquant.md,
    # FluxParams is defined and used.
    # Ensure the import of FluxParams (above) is correct relative to model_factory.py's location.

    params = FluxParams(
        # Provided by SVD_MODEL_ARCH_PARAMS in the script
        hidden_size=hidden_size,
        num_heads=num_heads,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        # Deduced/common values for Chroma/Flux.1-schnell
        in_channels=64,
        vec_in_dim=64,
        context_in_dim=4096,
        mlp_ratio=4.0,
        axes_dim=[
            64,
            64,
        ],  # hidden_size // num_heads = 3072 // 24 = 128. So [64,64] sums to 128.
        theta=10000,
        qkv_bias=True,
        guidance_embed=True,
    )
    flux_model = FluxMod(params=params, operations=nn)

    # Instantiate and assign the distilled_guidance_layer (Approximator)
    # Parameters from docs/chroma_model_layout_for_svdquant.md
    approximator_in_dim = 64
    approximator_hidden_dim = 5120
    approximator_out_dim = 3072  # Should match FluxParams.hidden_size
    approximator_n_layers = 5

    # Ensure out_dim of approximator matches hidden_size of main model
    if approximator_out_dim != params.hidden_size:
        raise ValueError(
            f"Approximator out_dim ({approximator_out_dim}) must match FluxParams hidden_size ({params.hidden_size})"
        )

    flux_model.distilled_guidance_layer = Approximator(
        in_dim=approximator_in_dim,
        out_dim=approximator_out_dim,
        hidden_dim=approximator_hidden_dim,
        n_layers=approximator_n_layers,
        operations=nn,
    )

    return flux_model
