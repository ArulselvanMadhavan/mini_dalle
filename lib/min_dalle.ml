type t =
  { models_root : string
  ; dtype : [ `f32 | `f16 ] option
  ; device : int option
  ; is_mega : bool
  ; is_reusable : bool option
  ; is_verbose : bool option
  ; layer_count : int
  ; text_token_count : int
  ; attention_head_count : int
  ; embed_count : int
  ; glu_embed_count : int
  ; text_vocab_count : int
  ; image_vocab_count : int
  ; vocab_path: string
  ; merges_path: string
  ; encoder_params_path : string
  ; decoder_params_path : string
  ; detoker_params_path : string
  }

type 'a with_config =
  ?models_root:string
  -> ?dtype:[ `f32 | `f16 ]
  -> ?device:int
  -> ?is_mega:bool
  -> ?is_reusable:bool
  -> ?is_verbose:bool
  -> 'a

let check_and_create_dirs dalle_path vqgan_path =
  if Sys.file_exists dalle_path then () else Sys.mkdir dalle_path 777;
  if Sys.file_exists vqgan_path then () else Sys.mkdir vqgan_path 777
;;

let mk ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () : t =
  let is_mega = Option.value is_mega ~default:true in
  let layer_count = if is_mega then 24 else 12 in
  let text_token_count = 64 in
  let attention_head_count = if is_mega then 32 else 16 in
  let embed_count = if is_mega then 2048 else 1024 in
  let glu_embed_count = if is_mega then 4096 else 2730 in
  let text_vocab_count = if is_mega then 50272 else 50264 in
  let image_vocab_count = if is_mega then 16415 else 16384 in
  let model_name = "dalle_bart_" ^ if is_mega then "mega" else "mini" in
  let models_root = Option.value models_root ~default:"pretrained" in
  let dalle_path = Filename.concat models_root model_name in
  let vqgan_path = Filename.concat models_root "vqgan" in
  check_and_create_dirs dalle_path vqgan_path;
  let vocab_path = Filename.concat dalle_path "vocab.json" in
  let merges_path = Filename.concat dalle_path "merges.txt" in
  let encoder_params_path = Filename.concat dalle_path "encoder.pt" in
  let decoder_params_path = Filename.concat dalle_path "decoder.pt" in
  let detoker_params_path = Filename.concat vqgan_path "detoker.pt" in
  
  { models_root
  ; dtype
  ; device
  ; is_mega
  ; is_reusable
  ; is_verbose
  ; layer_count
  ; text_token_count
  ; attention_head_count
  ; embed_count
  ; glu_embed_count
  ; text_vocab_count
  ; image_vocab_count
  ; vocab_path
  ; merges_path
  ; encoder_params_path
  ; decoder_params_path
  ; detoker_params_path
  }
;;

let make ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () =
  let m = mk ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () in
  m
;;
