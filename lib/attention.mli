open Torch

type t

val make : head_count:int -> embed_count:int -> t

val forward
  :  keys:Tensor.t
  -> values:Tensor.t
  -> queries:Tensor.t
  -> attention_mask:Tensor.t
  -> Tensor.t
