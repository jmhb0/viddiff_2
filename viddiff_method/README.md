# Viddiff method 

## OpenAI API
Two oft he components use 

## server for retrieval frames 
The retriever calls CLIP. To avoid loading CLIP each time 

Run `python apis/clip_server.py &`, which uses [OpenClip](https://github.com/mlfoundations/open_clip) like this: 
```
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k")
```
And then calls the CLIP server. 

Creates `tmp` directory which saves images for the CLIP server. This is not the fastes way to do this, but for a smaller dataset it's manageable. Also the function automatically does embedding caching into ``
