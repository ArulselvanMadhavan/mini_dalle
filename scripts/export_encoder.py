import numpy as np
import torch
import torchvision
from min_dalle.models import DalleBartEncoder

# encoder = DalleBartEncoder(
#             attention_head_count = self.attention_head_count,
#             embed_count = self.embed_count,
#             glu_embed_count = self.glu_embed_count,
#             text_token_count = self.text_token_count,
#             text_vocab_count = self.text_vocab_count,
#             layer_count = self.layer_count,
#             device=self.device
#         ).to(self.dtype).eval()

is_mega = True
text_token_count = 64
layer_count = 24 if is_mega else 12
attention_head_count = 32 if is_mega else 16
embed_count = 2048 if is_mega else 1024
glu_embed_count = 4096 if is_mega else 2730
text_vocab_count = 50272 if is_mega else 50264
image_vocab_count = 16415 if is_mega else 16384
device = "cpu"

params = torch.load("pretrained/dalle_bart_mega/encoder.pt")
encoder = DalleBartEncoder(
            attention_head_count = attention_head_count,
            embed_count = embed_count,
            glu_embed_count = glu_embed_count,
            text_token_count = text_token_count,
            text_vocab_count = text_vocab_count,
            layer_count = layer_count,
            device=device
)
encoder.load_state_dict(params, strict=False)

nps = {}
for k, v in encoder.state_dict().items():
    np.save("extracts/encoder/{}.npy".format(k), v.numpy())
    # nps[k] = v.numpy()

# m = torchvision.models.resnet18(pretrained=True)
# nps = {}
# for k, v in m.state_dict().items():
#     nps[k] = v.numpy()
    
# np.savez('resnet18.npz', **nps)
