type t
type 'a with_config = ?models_root:string -> 'a

val make : (unit -> t) with_config
