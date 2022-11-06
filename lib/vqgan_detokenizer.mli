open Torch

type t

val make : Var_store.t -> params_path:string -> t
val forward : t -> is_seamless:bool -> Tensor.t -> Tensor.t
