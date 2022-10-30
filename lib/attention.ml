open Torch

type t =
  { head_count : int
  ; embed_count : int
  ; k_proj : Nn.t
  ; v_proj : Nn.t
  ; q_proj : Nn.t
  ; out_proj : Nn.t
  }

let make vs ~head_count ~embed_count =
  let k_proj =
    Layer.linear
      Var_store.(vs / "k_proj")
      ~use_bias:false
      ~input_dim:embed_count
      embed_count
  in
  let v_proj =
    Layer.linear
      Var_store.(vs / "v_proj")
      ~use_bias:false
      ~input_dim:embed_count
      embed_count
  in
  let q_proj =
    Layer.linear
      Var_store.(vs / "q_proj")
      ~use_bias:false
      ~input_dim:embed_count
      embed_count
  in
  let out_proj =
    Layer.linear
      Var_store.(vs / "out_proj")
      ~use_bias:false
      ~input_dim:embed_count
      embed_count
  in
  { head_count; embed_count; k_proj; v_proj; q_proj; out_proj }
;;
