open Torch

type t

val make
  :  vs:Torch.Var_store.t
  -> layer_count:int
  -> embed_count:int
  -> attention_head_count:int
  -> text_vocab_count:int
  -> text_token_count:int
  -> glu_embed_count:int
  -> t

val forward : t -> text_tokens:Torch.Tensor.t -> Torch.Tensor.t

module EncoderLayer : sig
  type t

  val make : Var_store.t -> embed_count:int -> head_count:int -> glu_embed_count:int -> t
end
