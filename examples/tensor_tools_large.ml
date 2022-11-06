(* open Base *)
open Mini_dalle

let load_cmd =
  let open Cmdliner in
  let dir_path =
    Arg.(
      required
      & pos 0 (some string) None
      & info [] ~docv:"DIRPATH" ~doc:"Path to the directory")
  in
  let fname =
    Arg.(required & pos 1 (some string) None & info [] ~docv:"FILENAME" ~doc:"Filename")
  in
  let doc = "Load npy files to ot" in
  let man = [ `S "DESCRIPTION"; `P "Convert a npy file to ot file" ] in
  ( Term.(const Serialize.load_model $ dir_path $ fname)
  , Cmd.info "load" ~sdocs:"" ~doc ~man )
;;

let () =
  let open Cmdliner in
  let ser_cmd =
    let dir_path =
      Arg.(
        required
        & pos 0 (some string) None
        & info [] ~docv:"DIRPATH" ~doc:"Path to the directory")
    in
    let fname =
      Arg.(required & pos 1 (some string) None & info [] ~docv:"FILENAME" ~doc:"Filename")
    in
    let doc = "Serialize npy files to ot" in
    let man = [ `S "DESCRIPTION"; `P "Convert a npy file to ot file" ] in
    ( Term.(const Serialize.serialize_model $ dir_path $ fname)
    , Cmd.info "serialize" ~sdocs:"" ~doc ~man )
  in
  let default_cmd = Term.(ret (const (`Help (`Pager, None)))) in
  let info =
    let doc = "tools for converting large npy files to pytorch archives without zip" in
    Cmd.info "tensor_tools_large" ~version:"0.0.1" ~sdocs:"" ~doc
  in
  let cmds = [ ser_cmd; load_cmd ] |> List.map (fun (cmd, info) -> Cmd.v info cmd) in
  let main_cmd = Cmd.group info ~default:default_cmd cmds in
  Cmd.eval main_cmd |> Caml.exit
;;
