open Torch

type t =
  { layer_count : int
  ; embed_count : int
  ; image_vocab_count : int
  }

module DecoderSelfAttention = struct
  type t = { attn : Attention.t }

  let make vs head_count embed_count =
    let attn = Attention.make vs ~head_count ~embed_count in
    { attn }
  ;;

  let forward t ~decoder_state ~attention_state ~attention_mask ~token_index =
    let keys = Layer.forward t.attn.k_proj decoder_state in
    let values = Layer.forward t.attn.v_proj decoder_state in
    let queries = Layer.forward t.attn.q_proj decoder_state in
    let token_count = (Array.of_list (Tensor.shape token_index)).(1) in
    let keys, values, attention_state =
      if token_count == 1
      then (
        let batch_count = (Array.of_list (Tensor.shape decoder_state)).(0) in
        let attn_state_new = Tensor.concat [ keys; values ] ~dim:0 in
        let attn_state_new =
          Tensor.to_dtype
            attn_state_new
            ~dtype:(Tensor.type_ attention_state)
            ~non_blocking:true
            ~copy:false
        in
        let token_index_0 =
          Tensor.index token_index ~indices:[ Some (Tensor.of_int0 0) ]
        in
        let attention_state =
          Tensor.index_put
            attention_state
            ~indices:[ None; Some token_index_0 ]
            ~values:attn_state_new
            ~accumulate:false
        in
        let keys =
          Tensor.slice attention_state ~dim:0 ~start:None ~end_:(Some batch_count) ~step:1
        in
        let values =
          Tensor.slice attention_state ~dim:0 ~start:(Some batch_count) ~end_:None ~step:1
        in
        keys, values, attention_state)
      else keys, values, attention_state
    in
    let decoder_state = Attention.forward t.attn ~keys ~values ~queries ~attention_mask in
    decoder_state, attention_state
  ;;
end

module DecoderLayer = struct
  type t = { pre_self_attn_layer_norm : Nn.t }

  let make vs ~head_count ~embed_count ~glu_embed_count ~device =
    let pre_self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "pre_self_attn_layer_norm") embed_count
    in
    { pre_self_attn_layer_norm }
  ;;
end

let make
  vs
  ~image_vocab_count
  ~embed_count
  ~attention_head_count
  ~glu_embed_count
  ~layer_count
  ~device
  =
  List.iter print_int [ attention_head_count; glu_embed_count ];
  let embed_tokens =
    Layer.embeddings
      Var_store.(vs / "embed_tokens")
      ~num_embeddings:(image_vocab_count + 1)
      ~embedding_dim:embed_count
  in
  let embed_positions =
    Layer.embeddings
      Var_store.(vs / "embed_positions")
      ~num_embeddings:(Min_dalle.image_token_count + 1)
      ~embedding_dim:embed_count
  in
  { layer_count; embed_count; image_vocab_count }
;;
