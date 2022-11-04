import numpy as np
import torch
import torchvision
from min_dalle.models import VQGanDetokenizer
import os

EXTRACTS_PATH = "extracts/detokermega"

if not os.path.isdir(EXTRACTS_PATH):
    print('The directory is not present. Creating a new one..')
    os.mkdir(EXTRACTS_PATH)
else:
    print('The directory is present.')

is_mega = True
text_token_count = 64
layer_count = 24 if is_mega else 12
attention_head_count = 32 if is_mega else 16
embed_count = 2048 if is_mega else 1024
glu_embed_count = 4096 if is_mega else 2730
text_vocab_count = 50272 if is_mega else 50264
image_vocab_count = 16415 if is_mega else 16384
device = "cpu"

params = torch.load("pretrained/dalle_bart_mega/detoker.pt")

detoker = VQGanDetokenizer()

detoker.load_state_dict(params, strict=False)
nps = {}
for k,v in detoker.state_dict().items():
    np.save(EXTRACTS_PATH+"/{}.npy".format(k), v.numpy())
