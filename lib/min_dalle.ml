open Cohttp_lwt_unix
open Cohttp
open Lwt

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
  ; vocab_path : string
  ; merges_path : string
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

let exec_command cmd =
  let result = Sys.command cmd in
  if result != 0 then
    let err = Sys_error (cmd ^ " command failed") in
    raise err
  else
    ()

let mkdir_with_path = "mkdir -p "
  
let check_and_create_dirs dalle_path vqgan_path =
  if Sys.file_exists dalle_path then () else exec_command @@ mkdir_with_path ^ dalle_path;
  if Sys.file_exists vqgan_path then () else exec_command @@ mkdir_with_path ^ vqgan_path
;;

let min_dalle_repo = "https://huggingface.co/kuprel/min-dalle/resolve/main/"
let image_token_count = 256

let download_tokenizer m =
  let is_downloaded = Sys.file_exists m.vocab_path && Sys.file_exists m.merges_path in
  Printf.printf "\nis_downloaded:%B\n" is_downloaded
  
    
  
let init_tokenizer m =
  Client.get (Uri.of_string @@ min_dalle_repo ^ "config.json")
  >>= fun (resp, _body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code != 200 then
    Lwt.fail_with "HF config.json is not reachable"
  else
    Lwt.return @@ download_tokenizer m
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
  print_string m.detoker_params_path;
  print_string m.decoder_params_path;
  print_string m.encoder_params_path;
  print_string m.merges_path;
  print_string m.vocab_path;
  print_int m.image_vocab_count;
  print_int m.text_token_count;
  print_int m.glu_embed_count;
  print_int m.embed_count;
  print_int m.attention_head_count;
  print_int m.text_token_count;
  print_int m.layer_count;
  print_int m.text_vocab_count;
  Printf.printf "%B\n" @@ Option.value m.is_verbose ~default:true;
  Printf.printf "%B\n" @@ Option.value m.is_reusable ~default:true;
  Printf.printf "%B\n" m.is_mega;
  print_int @@ Option.value m.device ~default:0;
  print_int image_token_count;
  print_string m.models_root;
  (match Option.value m.dtype ~default:(`f32) with
  | `f16 -> print_string "f16"
  | `f32 -> print_string "f32");
  let tok_init = init_tokenizer m in
  tok_init >|= fun _body -> m
;;