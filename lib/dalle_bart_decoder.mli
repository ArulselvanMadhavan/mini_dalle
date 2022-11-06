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
  -> params_path:string
  -> t

val sample_tokens
  :  t
  -> settings:Tensor.t
  -> attention_mask:Tensor.t
  -> encoder_state:Tensor.t
  -> attention_state:Tensor.t
  -> prev_tokens:Tensor.t
  -> token_index:Tensor.t
  -> Tensor.t * Tensor.t
