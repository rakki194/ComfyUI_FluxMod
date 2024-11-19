import os
import json
import torch
import folder_paths

from .loader import load_flux_mod

class FluxModCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "guidance_name": (folder_paths.get_filename_list("checkpoints"),),
                "quant_mode": (["bf16", "float8_e4m3fn (8 bit)", "float8_e5m2 (also 8 bit)"],),
            }
        }
    

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/FluxMod"
    TITLE = "FluxModCheckpointLoader"

    def load_checkpoint(self, ckpt_name, guidance_name, quant_mode):
        dtypes = {
            "bf16": torch.bfloat16, 
            "float8_e4m3fn (8 bit)": torch.float8_e4m3fn, 
            "float8_e5m2 (also 8 bit)": torch.float8_e5m2
        }
            
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        guidance_path = folder_paths.get_full_path("checkpoints", guidance_name)
        flux_mod = load_flux_mod(
            model_path = ckpt_path,
            timestep_guidance_path = guidance_path,
            linear_dtypes=dtypes[quant_mode]
        )
        return (flux_mod,)

NODE_CLASS_MAPPINGS = {
    "FluxModCheckpointLoader" : FluxModCheckpointLoader,
}
