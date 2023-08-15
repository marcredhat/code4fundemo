
```
%%capture
!pip install jax
!pip install --upgrade pip # To support manylinux2010 wheels.
!pip install --upgrade jax jaxlib # CPU-only

!pip install flax
!pip install --upgrade diffusers transformers scipy 
!pip install torch 
#!pip -q uninstall diffusers
!pip install diffusers
!pip install transformers
!pip install diffusers["torch"] transformers
import torch
from diffusers import StableDiffusionPipeline
```


```
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

#prompt = "a photo of an astronaut riding a horse on mars"
#prompt = "a monkey eating a cactus with cherries"
#prompt = "kids coding for fun"
#prompt = "eating popcorn in a green jeep"
#prompt = "flying on a broom in Normandy"

#****1****
#prompt = "witch with blonde hair and red goggles"
#image = pipe(prompt).images[0]  
#image.save("witch.png")

#****2****
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
image.save("astronaut.png")


#****4****
#prompt = "a monkey eating a cactus with cherries"
#image = pipe(prompt).images[0]  
#image.save("monkey.png")

```
