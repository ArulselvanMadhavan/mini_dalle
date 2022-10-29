type t =
  { text_vocab_count : int
  ; embed_tokens : Torch.Nn.t
  ; embed_positions : Torch.Nn.t
  }

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

let forward t ~text_tokens =
  print_int t.text_vocab_count;
  Torch.Layer.forward t.embed_tokens text_tokens
;;
