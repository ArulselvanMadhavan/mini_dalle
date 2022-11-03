open Torch

type t

val make
  :  Var_store.t
  -> image_vocab_count:int
  -> embed_count:int
  -> attention_head_count:int
  -> glu_embed_count:int
  -> layer_count:int
  -> device:Torch.Device.t
  -> t
