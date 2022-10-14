type t = { models_root : string option }
type 'a with_config = ?models_root:string -> 'a

let mk ?models_root () : t = { models_root }

let make ?models_root () =
  let m = mk ?models_root () in
  let mr = Option.fold ~none:"" ~some:(fun b -> b) m.models_root in
  print_string mr;
  m
;;
