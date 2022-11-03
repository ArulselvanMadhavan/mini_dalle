open Torch
    
type t =
  { layer_count : int
  ; embed_count : int
  ; image_vocab_count : int
  }

module DecoderLayer = struct
  type t = {pre_self_attn_layer_norm: Nn.t}

  let make vs ~head_count ~embed_count ~glu_embed_count ~device =
    let pre_self_attn_layer_norm = Layer.layer_norm Var_store.(vs / "pre_self_attn_layer_norm") embed_count in
    {pre_self_attn_layer_norm}
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
  List.iter print_int [attention_head_count; glu_embed_count];
  let embed_tokens = Layer.embeddings Var_store.(vs / "embed_tokens") ~num_embeddings:(image_vocab_count+1) ~embedding_dim:embed_count in
  let embed_positions = Layer.embeddings Var_store.(vs / "embed_positions") ~num_embeddings:(Min_dalle.image_token_count + 1) ~embedding_dim:embed_count in
  
  { layer_count; embed_count; image_vocab_count }
;;
