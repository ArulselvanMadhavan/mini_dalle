open Torch

let read_npy dir_path name =
  match Npy.read_copy (dir_path ^ "/" ^ name) with
  | Npy.P tensor ->
    (match Bigarray.Genarray.layout tensor with
     | Bigarray.C_layout ->
       Filename.remove_extension name, Torch.Tensor.of_bigarray tensor
     | Bigarray.Fortran_layout -> failwith "fortran layout not supported")
;;

let load_model dir_path filename =
  let vs = Var_store.create ~name:"min-dalle" () in
  let _final_embed = Layer.layer_norm Var_store.(vs / "layers" // 0) 2048 in
  let named_tensors = Var_store.all_vars vs in
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
