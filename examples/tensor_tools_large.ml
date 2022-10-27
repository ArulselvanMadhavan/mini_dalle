(* open Base *)
open! Torch

let read_npy dir_path name =
  match Npy.read_copy (dir_path ^ "/" ^ name) with
  | Npy.P tensor ->
    (match Bigarray.Genarray.layout tensor with
     | Bigarray.C_layout ->
       Filename.remove_extension name, Torch.Tensor.of_bigarray tensor
     | Bigarray.Fortran_layout -> failwith "fortran layout not supported")
;;

let print_named_tensors xs =
  List.iter (fun (name, t) -> Stdio.printf "%s|%s\n" name @@ Tensor.shape_str t) xs
;;

let load_model dir_path filename =
  let vs = Var_store.create ~name:"min-dalle" () in
  let _final_embed = Layer.layer_norm Var_store.(vs / "final_ln") 2048 in
  let named_tensors = Var_store.all_vars vs in
  print_named_tensors named_tensors;
  Serialize.load_multi_ ~named_tensors ~filename:(dir_path ^ "/" ^ filename)
;;

let serialize_model dir_path filename =
  let named_tensors =
    Sys.readdir dir_path
    |> Array.to_list
    |> List.filter (fun x -> Filename.extension x = ".npy")
    |> List.map (read_npy dir_path)
  in
  Torch.Serialize.save_multi ~named_tensors ~filename:(dir_path ^ "/" ^ filename)
;;

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
  Term.(const load_model $ dir_path $ fname), Cmd.info "load" ~sdocs:"" ~doc ~man
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
    ( Term.(const serialize_model $ dir_path $ fname)
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
