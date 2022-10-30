open Torch

module EncoderLayer = struct
  type t =
    { pre_self_attn_layer_norm : Nn.t
    ; self_attn_layer_norm : Nn.t
    }

  let make vs ~embed_count ~head_count ~glu_embed_count =
    List.iter print_int [ head_count; glu_embed_count ];
    let pre_self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "pre_self_attn_layer_norm") embed_count
    in
    let self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "self_attn_layer_norm") embed_count
    in
    { pre_self_attn_layer_norm; self_attn_layer_norm }
  ;;

  let forward t encoder_state attn_mask =
    let residual = encoder_state in
    let encoder_state = Layer.forward t.pre_self_attn_layer_norm encoder_state in
    let encoder_state = Layer.forward t.self_attn_layer_norm encoder_state in
    let encoder_state = Tensor.( + ) encoder_state residual in
    Tensor.( * ) encoder_state attn_mask
  ;;
end

type t =
  { embed_tokens : Nn.t
  ; embed_positions : Nn.t
  ; layers : EncoderLayer.t list
  ; layernorm_embedding : Nn.t
  ; final_ln : Nn.t
  ; pose_tokens : Tensor.t
  }

let print_named_tensors xs =
  List.iter (fun (name, t) -> Stdio.printf "%s|%s\n" name @@ Tensor.shape_str t) xs
;;

let make
  ~vs
  ~layer_count
  ~embed_count
  ~attention_head_count
  ~text_vocab_count
  ~text_token_count
  ~glu_embed_count
  ~device
  =
  let embed_tokens =
    Layer.embeddings vs ~num_embeddings:embed_count ~embedding_dim:text_vocab_count
  in
  let embed_positions =
    Layer.embeddings vs ~num_embeddings:embed_count ~embedding_dim:text_token_count
  in
  let layernorm_embedding = Layer.layer_norm vs embed_count in
  let final_ln = Layer.layer_norm vs embed_count in
  let token_indices =
    Tensor.arange ~end_:(Scalar.int layer_count) ~options:(T Int, device)
  in
  let pose_tokens = Tensor.stack [ token_indices; token_indices ] ~dim:0 in
  let layers =
    List.init layer_count (fun x ->
      EncoderLayer.make
        Torch.Var_store.(vs / "layers" // x)
        ~embed_count
        ~head_count:attention_head_count
        ~glu_embed_count)
  in
  print_named_tensors @@ Var_store.all_vars vs;
  { embed_tokens; embed_positions; layers; layernorm_embedding; final_ln; pose_tokens }
;;

let forward t ~text_tokens =
  let attn_mask = Tensor.not_equal text_tokens (Scalar.i 1) in
  let t_forward = Layer.forward t.embed_tokens text_tokens in
  let p_forward = Layer.forward t.embed_positions t.pose_tokens in
  let encoder_state = Tensor.( + ) t_forward p_forward in
  let encoder_state = Layer.forward t.layernorm_embedding encoder_state in
  let encoder_state =
    List.fold_left
      (fun acc l -> EncoderLayer.forward l acc attn_mask)
      encoder_state
      t.layers
  in
  Layer.forward t.final_ln encoder_state
;;
