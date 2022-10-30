open Torch

type t =
  { head_count : int
  ; embed_count : int
  ; k_proj : Nn.t
  ; v_proj : Nn.t
  ; q_proj : Nn.t
  ; out_proj : Nn.t
  }

val make : Var_store.t -> head_count:int -> embed_count:int -> t

val forward
  :  t
  -> keys:Tensor.t
  -> values:Tensor.t
  -> queries:Tensor.t
  -> attention_mask:Tensor.t
  -> Tensor.t
