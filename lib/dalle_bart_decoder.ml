open Torch

module DecoderCrossAttention = struct
  type t = { attn : Attention.t }

  let make vs head_count embed_count =
    let attn = Attention.make vs ~head_count ~embed_count in
    { attn }
  ;;

  let forward t ~decoder_state ~encoder_state ~attention_mask =
    let keys = Layer.forward t.attn.k_proj encoder_state in
    let values = Layer.forward t.attn.v_proj encoder_state in
    let queries = Layer.forward t.attn.q_proj decoder_state in
    Attention.forward t.attn ~keys ~values ~queries ~attention_mask
  ;;
end

let token_count token_index = (Array.of_list (Tensor.shape token_index)).(1)

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
    let token_count = token_count token_index in
    let keys, values, attention_state =
      if token_count == 1
      then (
        let batch_count = List.hd (Tensor.shape decoder_state) in
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
  type t =
    { pre_self_attn_layer_norm : Nn.t
    ; self_attn : DecoderSelfAttention.t
    ; self_attn_layer_norm : Nn.t
    ; pre_encoder_attn_layer_norm : Nn.t
    ; encoder_attn : DecoderCrossAttention.t
    ; encoder_attn_layer_norm : Nn.t
    ; glu : Glu.t
    ; token_indices : Tensor.t
    }

  let make vs ~head_count ~embed_count ~glu_embed_count ~device =
    let pre_self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "pre_self_attn_layer_norm") embed_count
    in
    let self_attn =
      DecoderSelfAttention.make Var_store.(vs / "self_attn") head_count embed_count
    in
    let self_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "self_attn_layer_norm") embed_count
    in
    let pre_encoder_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "pre_encoder_attn_layer_norm") embed_count
    in
    let encoder_attn =
      DecoderCrossAttention.make Var_store.(vs / "encoder_attn") head_count embed_count
    in
    let encoder_attn_layer_norm =
      Layer.layer_norm Var_store.(vs / "encoder_attn_layer_norm") embed_count
    in
    let glu =
      Glu.make
        Var_store.(vs / "glu")
        ~count_in_out:embed_count
        ~count_middle:glu_embed_count
    in
    let token_indices =
      Tensor.arange ~end_:(Scalar.i Constants.image_token_count) ~options:(T Int64, device)
    in
    { pre_self_attn_layer_norm
    ; self_attn
    ; self_attn_layer_norm
    ; pre_encoder_attn_layer_norm
    ; encoder_attn
    ; encoder_attn_layer_norm
    ; glu
    ; token_indices
    }
  ;;

  let forward
    t
    ~decoder_state
    ~encoder_state
    ~attention_state
    ~attention_mask
    ~token_index
    =
    (* self attention *)
    let token_count = token_count token_index in
    let self_attn_mask =
      if token_count == 1
      then (
        let self_attn_mask = Tensor.less_equal_tensor t.token_indices token_index in
        let shp = Tensor.shape self_attn_mask in
        Tensor.reshape self_attn_mask ~shape:[ List.hd shp; 1; 1; Base.List.last_exn shp ])
      else (
        (* token_indices[:, :, :token_count] *)
        let tind = Tensor.unsqueeze t.token_indices ~dim:0 in
        let tind = Tensor.unsqueeze tind ~dim:1 in
        let tind =
          Tensor.slice tind ~dim:2 ~start:None ~end_:(Some token_count) ~step:1
        in
        (*  token_index[:, :, None]*)
        let shp = Array.of_list (Tensor.shape token_index) in
        let tind2 = Tensor.reshape token_index ~shape:[ shp.(0); shp.(1); 1 ] in
        let self_attn_mask = Tensor.less_equal_tensor tind tind2 in
        Tensor.unsqueeze ~dim:1 self_attn_mask)
    in
    let residual = decoder_state in
    let decoder_state = Layer.forward t.pre_self_attn_layer_norm decoder_state in
    let decoder_state, attention_state =
      DecoderSelfAttention.forward
        t.self_attn
        ~decoder_state
        ~attention_state
        ~attention_mask:self_attn_mask
        ~token_index
    in
    let decoder_state = Layer.forward t.self_attn_layer_norm decoder_state in
    let decoder_state = Tensor.(residual + decoder_state) in
    (* Cross Attention *)
    let residual = decoder_state in
    let decoder_state = Layer.forward t.pre_encoder_attn_layer_norm decoder_state in
    let decoder_state =
      DecoderCrossAttention.forward
        t.encoder_attn
        ~decoder_state
        ~encoder_state
        ~attention_mask
    in
    let decoder_state = Layer.forward t.encoder_attn_layer_norm decoder_state in
    let decoder_state = Tensor.(residual + decoder_state) in
    (* Feed forward *)
    let residual = decoder_state in
    let decoder_state = Glu.forward t.glu decoder_state in
    let decoder_state = Tensor.(residual + decoder_state) in
    decoder_state, attention_state
  ;;
end

type t =
  { embed_tokens : Nn.t
  ; embed_positions : Nn.t
  ; layers : DecoderLayer.t list
  ; layernorm_embedding : Nn.t
  ; final_ln : Nn.t
  ; lm_head : Nn.t
  }

(* let print_named_tensors = *)
(*   List.iteri (fun idx (name, t) -> *)
(*     Stdio.printf "%d)%s:%s\n" idx name (Tensor.shape_str t)) *)

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
      ~num_embeddings:Constants.image_token_count
      ~embedding_dim:embed_count
  in
  let layers =
    List.init layer_count (fun x ->
      DecoderLayer.make
        Var_store.(vs / "layers" // x)
        ~head_count:attention_head_count
        ~embed_count
        ~glu_embed_count
        ~device)
  in
  let layernorm_embedding =
    Layer.layer_norm Var_store.(vs / "layernorm_embedding") embed_count
  in
  let final_ln = Layer.layer_norm Var_store.(vs / "final_ln") embed_count in
  let lm_head =
    Layer.linear
      Var_store.(vs / "lm_head")
      ~use_bias:false
      ~input_dim:embed_count
      (image_vocab_count + 1)
  in
  Serialize.load_multi_
    ~named_tensors:(Var_store.all_vars vs)
    ~filename:"extracts/decodermega/decoder.ot";
  Stdio.printf "*****Decoder load complete*****\n";  
  (* print_named_tensors (Var_store.all_vars vs); *)
  { embed_tokens; embed_positions; layers; layernorm_embedding; final_ln; lm_head }
;;

let forward t ~attention_mask ~encoder_state ~attention_state ~prev_tokens ~token_index =
  let image_count = Int.div (List.hd (Tensor.shape encoder_state)) 2 in
  let token_index = Tensor.unsqueeze token_index ~dim:0 in
  let token_index = Tensor.repeat token_index ~repeats:[ image_count * 2; 1 ] in
  let prev_tokens = Tensor.repeat prev_tokens ~repeats:[ 2; 1 ] in
  let decoder_state = Layer.forward t.embed_tokens prev_tokens in
  let pos_enc = Layer.forward t.embed_positions token_index in
  let decoder_state = Tensor.(decoder_state + pos_enc) in
  let decoder_state = Layer.forward t.layernorm_embedding decoder_state in
  let decoder_state, attention_state =
    Base.List.foldi
      t.layers
      ~init:(decoder_state, attention_state)
      ~f:(fun idx (decoder_state, attention_state) l ->
      let idx = Tensor.of_int0 idx in
      let att_state = Tensor.index attention_state ~indices:[ Some idx ] in
      let decoder_state, att_state =
        DecoderLayer.forward
          l
          ~decoder_state
          ~encoder_state
          ~attention_state:att_state
          ~attention_mask
          ~token_index
      in
      let attention_state =
        Tensor.index_put_
          attention_state
          ~indices:[ Some idx ]
          ~values:att_state
          ~accumulate:false
      in
      decoder_state, attention_state)
  in
  let decoder_state = Layer.forward t.final_ln decoder_state in
  let logits = Layer.forward t.lm_head decoder_state in
  logits, attention_state
;;

let sample_tokens
  t
  ~settings
  ~attention_mask
  ~encoder_state
  ~attention_state
  ~prev_tokens
  ~token_index
  =
  let logits, attention_state =
    forward t ~attention_mask ~encoder_state ~attention_state ~prev_tokens ~token_index
  in
  let image_count = Int.div (List.hd (Tensor.shape logits)) 2 in
  let temperature =
    Tensor.to_float0_exn (Tensor.index settings ~indices:[ Some (Tensor.of_int0 0) ])
  in
  let top_k =
    Tensor.to_float0_exn (Tensor.index settings ~indices:[ Some (Tensor.of_int0 1) ])
  in
  let supercondition_factor =
    Tensor.to_float0_exn (Tensor.index settings ~indices:[ Some (Tensor.of_int0 2) ])
  in
  let logits = Tensor.index logits ~indices:[ None; Some (Tensor.of_int0 (-1)); None ] in
  let logits = Tensor.unsqueeze ~dim:1 logits in
  let logits =
    Tensor.slice ~dim:2 ~start:None ~end_:(Some (Base.Int.pow 2 14)) ~step:1 logits
  in
  let diversity_logits =
    Tensor.slice ~dim:0 ~start:None ~end_:(Some image_count) ~step:1 logits
  in
  let relevancy_logits =
    Tensor.slice ~dim:0 ~start:(Some image_count) ~end_:None ~step:1 logits
  in
  let diversity_logits =
    Tensor.(mul_scalar diversity_logits (Scalar.f (1. -. supercondition_factor)))
  in
  let relevancy_logits =
    Tensor.(mul_scalar relevancy_logits (Scalar.f supercondition_factor))
  in
  let logits = Tensor.(diversity_logits + relevancy_logits) in
  let logits = Tensor.squeeze logits in
  let logits_sorted, _ = Tensor.sort ~dim:(-1) logits ~descending:true in
  let logits_sorted_topk =
    Tensor.index
      logits_sorted
      ~indices:[ None; Some (Tensor.of_int0 (Int.of_float top_k - 1)) ]
  in
  let logits_sorted_topk = Tensor.unsqueeze ~dim:1 logits_sorted_topk in
  let is_kept =
    Tensor.to_dtype
      (Tensor.greater_equal_tensor logits logits_sorted_topk)
      ~dtype:(T Float)
      ~copy:false
      ~non_blocking:true
  in
  let logits_top =
    Tensor.index logits_sorted ~indices:[ None; Some (Tensor.of_int0 0) ]
  in
  let logits_top = Tensor.unsqueeze ~dim:1 logits_top in
  let logits = Tensor.(logits - logits_top) in
  let logits = Tensor.(div_scalar logits (Scalar.f temperature)) in
  let logits = Tensor.exp_ logits in
  let logits = Tensor.mul logits is_kept in
  let image_tokens = Tensor.multinomial logits ~num_samples:1 ~replacement:false in
  image_tokens, attention_state
;;
