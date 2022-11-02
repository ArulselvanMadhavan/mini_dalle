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

let forward t ~keys ~values ~queries ~attention_mask =
  (* Assumes FP32 *)
  let k_shape = List.append (Base.List.take (Tensor.shape keys) 2) [ t.head_count; -1 ] in
  let v_shape =
    List.append (Base.List.take (Tensor.shape values) 2) [ t.head_count; -1 ]
  in
  let q_shape =
    List.append (Base.List.take (Tensor.shape queries) 2) [ t.head_count; -1 ]
  in
  let keys = Tensor.reshape keys ~shape:k_shape in
  let values = Tensor.reshape values ~shape:v_shape in
  let queries = Tensor.reshape queries ~shape:q_shape in
  let q_div = Float.sqrt @@ Float.of_int @@ Base.List.last_exn @@ Tensor.shape queries in
  let queries = Tensor.div_scalar queries (Scalar.f q_div) in
  let attention_mask =
    Tensor.to_dtype attention_mask ~dtype:(T Float) ~non_blocking:true ~copy:false
  in
  let attention_bias = Tensor.(add_scalar (neg attention_mask) (Scalar.int 1)) in
  let attention_bias = Tensor.(mul_scalar attention_bias (Scalar.float (-1e12))) in
  let attention_weights = Tensor.einsum ~equation:"bqhc,bkhc->bhqk" [ queries; keys ] in
  let attention_weights = Tensor.(attention_weights + attention_bias) in
  let attention_weights = Tensor.softmax attention_weights ~dim:(-1) ~dtype:(T Float) in
  let attention_output =
    Tensor.einsum ~equation:"bhqk,bkhc->bqhc" [ attention_weights; values ]
  in
  let shape =
    List.append (Base.List.take (Tensor.shape attention_output) 2) [ t.embed_count ]
  in
  let attention_output = Tensor.reshape attention_output ~shape in
  Layer.forward t.out_proj attention_output
;;
