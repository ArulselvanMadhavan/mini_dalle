type t

val make : (string, int) Hashtbl.t -> string list -> t

val token_count : t -> int

val pairs_count : t -> int
