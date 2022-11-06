open Torch

type t

type 'a with_config =
  ?models_root:string
  -> ?dtype:[ `f32 | `f16 ]
  -> ?device:int
  -> ?is_mega:bool
  -> ?is_reusable:bool
  -> ?is_verbose:bool
  -> 'a

val make : (unit -> t Lwt.t) with_config

val generate_raw_image_stream
  :  text:string
  -> seed:int
  -> grid_size:int
  -> ?is_seamless:bool
  -> ?temperature:float
  -> ?top_k:int
  -> ?supercondition_factor:int
  -> t
  -> Tensor.t

val fetch_file : string -> bool -> bool -> string -> unit Lwt.t
