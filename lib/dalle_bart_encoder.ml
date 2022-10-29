type t =
  { text_vocab_count : int
  ; embed_tokens : Torch.Nn.t
  ; embed_positions : Torch.Nn.t
  }

module EncoderLayer = struct
  type t =
    { pre_self_attn_layer_norm : Torch.Nn.t
    ; self_attn_layer_norm : Torch.Nn.t
    }

  let make vs embed_count head_count glu_embed_count =
    List.iter print_int [ head_count; glu_embed_count ];
    let pre_self_attn_layer_norm = Torch.Layer.layer_norm vs embed_count in
    let self_attn_layer_norm = Torch.Layer.layer_norm vs embed_count in
    { pre_self_attn_layer_norm; self_attn_layer_norm }
  ;;
end

let make
  ~layer_count
  ~embed_count
  ~attention_head_count
  ~text_vocab_count
  ~text_token_count
  ~glu_embed_count
  ~vs
  =
  List.iter
    print_int
    [ layer_count
    ; embed_count
    ; attention_head_count
    ; text_vocab_count
    ; text_token_count
    ; glu_embed_count
    ];
  let embed_tokens =
    Torch.Layer.embeddings vs ~num_embeddings:embed_count ~embedding_dim:text_vocab_count
  in
  let embed_positions =
    Torch.Layer.embeddings vs ~num_embeddings:embed_count ~embedding_dim:text_token_count
  in
  { text_vocab_count; embed_tokens; embed_positions }
;;

let forward t ~text_tokens = Torch.Layer.forward t.embed_tokens text_tokens
