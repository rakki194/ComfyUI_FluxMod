# ComfyUI FluxMod 🚀

A modulation layer addon for Flux that reduces model size to 8.8B parameters without significant quality loss.

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Modulation%20Layer-yellow)](https://huggingface.co/lodestone-horizon/flux-essence)

## Overview

ComfyUI_FluxMod acts as a plugin for Flux, enabling you to run Flux Dev and Flux Schnell on more consumer-friendly hardware. This is achieved by utilizing a modulation layer that significantly reduces the parameter count while maintaining quality.

> **Note**: You still need the original Flux Dev or Flux Schnell model to use this addon.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Node Information](#node-information)
- [Quantization Guide](#quantization-guide)
- [FAQ](#faq)
- [Support](#support)
- [Contributing](#contributing)
- [Issues](#issues)
- [Sample Workflow](#sample-workflow)

## Requirements

- ComfyUI installation
- Original Flux model (Dev or Schnell)
- [universal_modulator.safetensors](https://huggingface.co/lodestone-horizon/flux-essence)

## Installation

It's recommended to use either ComfyUI Manager or Comfy Registry.

### Via ComfyUI Manager (GUI)
1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if you haven't already
2. Open ComfyUI and click on the Manager Button (puzzle piece icon)
3. Go to "Custom Nodes Manager" tab
4. Search for "ComfyUI_FluxMod"
5. Click Install
6. Restart ComfyUI

### Via Comfy Registry (CLI)

```bash
comfy node registry-install comfyui_fluxmod
```

### Manual Installation

1. Navigate to your ComfyUI's custom_nodes folder
2. Clone the repository:

```bash
git clone https://github.com/lodestone-rock/ComfyUI_FluxMod.git
```

3. Restart ComfyUI
4. Refresh your browser if ComfyUI is already running

## Node Information

| Node                    | Description                     | Options                                                                                                                             |
| ----------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| FluxModCheckpointLoader | Primary checkpoint loading node | • **ckpt_name**: Original Flux model path<br>• **guidance_name**: Modulation addon path<br>• **quant_mode**: Quantization selection |

## Usage

1. Double click workspace → search "FluxModCheckpointLoader"
2. Select your Flux model in `ckpt_name`
3. Select modulation addon in `guidance_name`
4. Choose quantization mode
5. Configure remaining nodes as per standard Flux workflow

> 💡 **Tip**: Check the `examples` folder for sample workflows. Drag and drop the workflow image into ComfyUI to get started quickly.

## Quantization Guide

| Mode          | Recommended GPU | VRAM Usage | Recommended |
| ------------- | --------------- | ---------- | ----------- |
| bf16          | 24GB+           | ~20GB      | ✅          |
| float8_e4m3fn | 12-16GB         | ~10GB      | ✅          |
| float8_e5m2   | 12-16GB         | ~10GB      | ❌          |

## FAQ

<details>
<summary><b>Will my outputs be different?</b></summary>
Yes, outputs will likely differ as we're reducing parameters. However, the difference is often minimal for most use cases.
</details>

<details>
<summary><b>How much does this degrade image quality?</b></summary>
Testing shows minimal quality degradation in most cases. The most notable exception is long text generation, which shows a moderate degradation.
</details>

<details>
<summary><b>Why use this over regular quantization?</b></summary>
You don't have to use this over regular quantisation! You can combine them, or if you don't want to use quantisation at all and you have enough VRAM, you can also just stick with bf16. If you combine quantisation, you can make the model even smaller and allow it to run on consumer hardware.
</details>

<details>
<summary><b>I tried exporting/saving this model and I got an error?</b></summary>
This model has a completely different architecture compared to the original Flux and none of the current methods for exporting/saving models would support it. This is why we needed to have this custom node created in the first place, since otherwise it wouldn't load properly.
</details>

## Support

Need help? Join our [Discord community](https://discord.gg/UxBAMcpqDU) for support and discussions.

## Contributing

Pull requests are welcome! Feel free to contribute to the project by:

- Fixing bugs
- Adding new features
- Improving documentation
- Suggesting enhancements

## Issues

Found a bug? Have a suggestion? Please create a [GitHub issue](https://github.com/lodestone-rock/ComfyUI_FluxMod/issues) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment details (OS, GPU, etc.)

## Sample Workflow

![Workflow Example](https://github.com/lodestone-rock/flux-mod/blob/main/examples/1.png)
