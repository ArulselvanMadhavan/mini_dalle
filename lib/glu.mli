open Torch

type t

val make : Var_store.t -> count_in_out:int -> count_middle:int -> t
val forward : t -> Tensor.t -> Tensor.t
