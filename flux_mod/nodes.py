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
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/FluxMod"
    TITLE = "FluxModCheckpointLoader"

    def load_checkpoint(self, ckpt_name, guidance_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        guidance_path = folder_paths.get_full_path("checkpoints", guidance_name)
        flux_mod = load_flux_mod(
            model_path = ckpt_path,
            timestep_guidance_path = guidance_path
        )
        return (flux_mod,)

NODE_CLASS_MAPPINGS = {
    "FluxModCheckpointLoader" : FluxModCheckpointLoader,
}
