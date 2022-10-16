type t

val make
  :  layer_count:int
  -> embed_count:int
  -> attention_head_count:int
  -> text_vocab_count:int
  -> text_token_count:int
  -> glu_embed_count:int
  -> vs:Torch.Var_store.t
  -> t

val forward : t -> text_tokens:Torch.Tensor.t -> Torch.Tensor.t
