type t =
  { token_from_subword : (string, int) Hashtbl.t
  ; rank_from_pair : (string * string, int) Hashtbl.t
  }

val make : (string, int) Hashtbl.t -> string list -> t
val tokenize : t -> text:string -> is_verbose:bool -> int list
