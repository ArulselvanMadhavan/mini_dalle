open Torch

module EncoderSelfAttention = struct
  type t = { attn : Attention.t }

  let make attn = { attn }

  let forward t encoder_state attn_mask =
    let keys = Layer.forward t.attn.k_proj encoder_state in
    let values = Layer.forward t.attn.v_proj encoder_state in
    let queries = Layer.forward t.attn.q_proj encoder_state in
    Attention.forward t.attn ~keys ~values ~queries ~attention_mask:attn_mask
  ;;
end

module EncoderLayer = struct
  type t =
    { pre_self_attn_layer_norm : Nn.t
    ; self_attn_layer_norm : Nn.t
    ; glu : Glu.t
    ; self_attn : EncoderSelfAttention.t
    }

  let make vs ~embed_count ~head_count ~glu_embed_count =
    let pre_self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "pre_self_attn_layer_norm") embed_count
    in
    let self_attn =
      EncoderSelfAttention.make
        (Attention.make Var_store.(vs / "self_attn") ~head_count ~embed_count)
    in
    let self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "self_attn_layer_norm") embed_count
    in
    let glu =
      Glu.make
        Var_store.(vs / "glu")
        ~count_in_out:embed_count
        ~count_middle:glu_embed_count
    in
    { pre_self_attn_layer_norm; self_attn_layer_norm; glu; self_attn }
  ;;

  let forward t encoder_state attn_mask =
    let residual = encoder_state in
    let encoder_state = Layer.forward t.pre_self_attn_layer_norm encoder_state in
    let encoder_state =
      EncoderSelfAttention.forward t.self_attn encoder_state attn_mask
    in
    let encoder_state = Layer.forward t.self_attn_layer_norm encoder_state in
    let encoder_state = Tensor.(residual + encoder_state) in
    let residual = encoder_state in
    let encoder_state = Glu.forward t.glu encoder_state in
    Tensor.(residual + encoder_state)
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

let make
  ~vs
  ~layer_count
  ~embed_count
  ~attention_head_count
  ~text_vocab_count
  ~text_token_count
  ~glu_embed_count
  ~device
  ~params_path
  =
  let embed_tokens =
    Layer.embeddings
      Var_store.(vs / "embed_tokens")
      ~num_embeddings:text_vocab_count
      ~embedding_dim:embed_count
  in
  let embed_positions =
    Layer.embeddings
      Var_store.(vs / "embed_positions")
      ~num_embeddings:text_token_count
      ~embedding_dim:embed_count
  in
  let layernorm_embedding =
    Layer.layer_norm Var_store.(vs / "layernorm_embedding") embed_count
  in
  let final_ln = Layer.layer_norm Var_store.(vs / "final_ln") embed_count in
  let token_indices =
    Tensor.arange ~end_:(Scalar.int text_token_count) ~options:(T Int64, device)
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
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:params_path;
  { embed_tokens; embed_positions; layers; layernorm_embedding; final_ln; pose_tokens }
;;

let forward t ~text_tokens =
  let attn_mask = Tensor.not_equal text_tokens (Scalar.i 1) in
  let mask_shp = Tensor.shape attn_mask in
  let attn_mask =
    Tensor.reshape
      attn_mask
      ~shape:[ List.hd mask_shp; 1; 1; Base.List.last_exn mask_shp ]
  in
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
