```
%%capture
#https://github.com/huggingface/blog/blob/main/openvino.md
!pip install pip --upgrade
!pip install optimum[openvino,nncf] torchvision evaluate
!pip install openvino
!pip install torchvision
!pip install evaluate
!pip install rawkit
!pip install rawpy
!pip install --upgrade-strategy eager optimum[openvino,nncf]
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import requests
import PIL
import io
from PIL import Image
from io import BytesIO
from PIL import Image, ImageFile
import numpy
from transformers import pipeline
from optimum.intel.openvino import OVQuantizer
from optimum.intel.openvino import OVQuantizer
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from optimum.intel.openvino import OVModelForImageClassification
import torch
```


```


model_id = "juliensimon/autotrain-food101-1471154050"
model = AutoModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
```


```
ov_model = OVModelForImageClassification.from_pretrained("echarlaix/vit-food101-int8")
ov_pipe = pipeline("image-classification", model=ov_model, feature_extractor=feature_extractor)

#from optimum.intel.openvino import OVModelForImageClassification
#ov_model = OVModelForImageClassification.from_pretrained(save_dir)
#feature_extractor.save_pretrained(save_dir)


#Marmiton BIBIMBAP
#https://www.marmiton.org/recettes/recette_bibimbap-coreen_340709.aspx

#outputs = ov_pipe("https://assets.afcdn.com/recipe/20151026/24405_w2000h2522c1cx1272cy1604.webp")
#print(outputs)


#Marmiton FRIED RICE
#https://www.marmiton.org/recettes/recette_le-vrai-riz-frit-chinois_44676.aspx
outputs = ov_pipe("https://assets.afcdn.com/recipe/20230104/139102_w1280h972c1cx1062cy705cxb2124cyb1411.webp")
print(outputs)


#Marmiton CHICKEN CURRY
#https://www.marmiton.org/recettes/recette_poulet-curry-et-oignons-facile_13026.aspx

#outputs = ov_pipe("https://assets.afcdn.com/recipe/20160405/6932_w2000h1331c1cx2125cy1414.webp")
#print(outputs)


#Marmiton RAMEN 
#outputs = ov_pipe("https://assets.afcdn.com/recipe/20191204/103426_w2000h1333c1cx2600cy2116cxb5760cyb3840.webp")


#byteImg = convert_cr2_to_jpg("https://upload.wikimedia.org/wikipedia/commons/c/c0/Cibo_nordafricano.jpg")
```
```````
