open Torch

type t =
  { fc0 : Nn.t
  ; fc1 : Nn.t
  ; fc2 : Nn.t
  ; ln0 : Nn.t
  ; ln1 : Nn.t
  }

let make vs ~count_in_out ~count_middle =
  let ln0 = Layer.layer_norm Var_store.(vs / "ln0") count_in_out in
  let ln1 = Layer.layer_norm Var_store.(vs / "ln1") count_in_out in
  let fc0 =
    Layer.linear
      Var_store.(vs / "fc0")
      ~use_bias:false
      ~input_dim:count_in_out
      count_middle
  in
  let fc1 =
    Layer.linear
      Var_store.(vs / "fc1")
      ~use_bias:false
      ~input_dim:count_in_out
      count_middle
  in
  let fc2 =
    Layer.linear
      Var_store.(vs / "fc2")
      ~use_bias:false
      ~input_dim:count_middle
      count_in_out
  in
  { fc0; fc1; fc2 }
;;

let forward z = z
