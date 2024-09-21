import socket 
NODE = socket.gethostname().split(".")[0]
PROJECT = "/pasteur/u/jmhb/viddiff_2"

# CLIP API
# CLIP_URL = "http://pasteur6:8090"
# CLIP_URL = f"http://{NODE}:8090"
CLIP_URL = f"http://pasteur5:8090"
CLIP_CACHE_FILE = f"cache/cache_clip"



