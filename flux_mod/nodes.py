import os
import re
import json
import torch
import folder_paths

from .loader import load_flux_mod
from .sampler import common_ksampler
import comfy.samplers 


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
            linear_dtypes=dtypes[quant_mode],
            lite_patch_path=None
        )
        return (flux_mod,)

class FluxModCheckpointLoaderMini:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "guidance_name": (folder_paths.get_filename_list("checkpoints"),),
                "quant_mode": (["bf16", "float8_e4m3fn (8 bit)", "float8_e5m2 (also 8 bit)"],),
                "lite_patch_ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }
    

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "ExtraModels/FluxMod"
    TITLE = "FluxModCheckpointLoaderMini"

    def load_checkpoint(self, ckpt_name, guidance_name, quant_mode, lite_patch_ckpt_name):
        dtypes = {
            "bf16": torch.bfloat16, 
            "float8_e4m3fn (8 bit)": torch.float8_e4m3fn, 
            "float8_e5m2 (also 8 bit)": torch.float8_e5m2
        }
            
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        guidance_path = folder_paths.get_full_path("checkpoints", guidance_name)
        lite_patch_ckpt_name = folder_paths.get_full_path("checkpoints", lite_patch_ckpt_name)
        flux_mod = load_flux_mod(
            model_path = ckpt_path,
            timestep_guidance_path = guidance_path,
            linear_dtypes=dtypes[quant_mode],
            lite_patch_path=lite_patch_ckpt_name
        )
        return (flux_mod,)


class ModelMover:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("MODEL", ),
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
    
class SkipLayerForward:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "skip_mmdit_layers": ("STRING", {"default": "10", "multiline": False}), 
                "skip_dit_layers": ("STRING", {"default": "3, 4", "multiline": False})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "skip_layer"
    CATEGORY = "ExtraModels/FluxMod"
    TITLE = "SkipLayerForward"

    DESCRIPTION = "Prune model layers"

    def skip_layer(self, model, skip_mmdit_layers, skip_dit_layers):
        
        skip_mmdit_layers = re.split(r"\s*,\s*", skip_mmdit_layers)
        skip_mmdit_layers = [int(num) for num in skip_mmdit_layers]

        skip_dit_layers = re.split(r"\s*,\s*", skip_dit_layers)
        skip_dit_layers = [int(num) for num in skip_dit_layers]

        model.model.diffusion_model.skip_dit = skip_dit_layers
        model.model.diffusion_model.skip_mmdit = skip_mmdit_layers
        return (model, )
    

class KSamplerMod:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "activation_casting": (["bf16", "fp16"], {"default": "bf16", "tooltip": "cast model activation to bf16 or fp16, always use bf16 unless your card does not supports it"})
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "ExtraModels/FluxMod"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."
    TITLE = "KSamplerMod"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, activation_casting="bf16"):
        dtypes = {
            "bf16": torch.bfloat16, 
            "fp16": torch.float16
        }
        with torch.autocast(device_type="cuda", dtype=dtypes[activation_casting]):
            return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
    
NODE_CLASS_MAPPINGS = {
    "FluxModCheckpointLoader" : FluxModCheckpointLoader,
    "FluxModCheckpointLoaderMini": FluxModCheckpointLoaderMini,
    "KSamplerMod": KSamplerMod,
    "SkipLayerForward": SkipLayerForward
}
