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
