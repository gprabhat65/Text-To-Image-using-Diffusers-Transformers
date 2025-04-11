# 🖼️ Text to Image using Diffusers & Transformers

This project transforms simple text prompts into stunning AI-generated images using the **Stable Diffusion v1.4** model, powered by Hugging Face's `diffusers` and `transformers` libraries. The implementation supports CUDA acceleration for high-performance image generation.

---

## 🔥 Features

- Text-to-Image generation using **Stable Diffusion**
- GPU acceleration via CUDA & PyTorch
- Simple CLI input prompt
- Image auto-save and display
- Lightweight and beginner-friendly

---

## 🛠️ Installation

> 💡 Requires Python 3.7+ and CUDA-compatible GPU

1. **Uninstall previous PyTorch versions** (optional but recommended):

```bash
pip uninstall -y torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install diffusers transformers accelerate pillow

from huggingface_hub import login
login(token="your_huggingface_token_here")

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import IPython.display as display

# Load model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.to(device)

# User prompt
prompt = input("Enter your prompt: ")

# Generate image
with autocast(device):
    image = pipe(prompt, guidance_scale=8.5).images[0]

# Save and show
image.save("GeneratedImage.png")
display.display(image)


📷 Output Example
Below is an example of a generated image from a sample prompt:


⚙️ Requirements
Python 3.7+

PyTorch with CUDA support

Hugging Face Account (for model access)

🧠 Model Used
CompVis/stable-diffusion-v1-4

🔮 Roadmap
 Add Streamlit or Gradio UI

 Batch prompt input

 Prompt history and logging

🙌 Credits
Hugging Face

Stable Diffusion by CompVis


Let me know if you'd like a version that includes a `.ipynb` or `.py` file reference or if you're deploying this on Hugging Face Spaces or GitHub Pages — I can tailor it further!



