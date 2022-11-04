open Torch

type t

val make : Var_store.t -> t
val forward : t -> is_seamless:bool -> Tensor.t -> Tensor.t
