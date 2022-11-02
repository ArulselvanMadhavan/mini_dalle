open Cohttp_lwt_unix
open Cohttp
open Lwt

type t =
  { models_root : string
  ; dtype : [ `f32 | `f16 ] option
  ; device : Torch_core.Device.t
  ; is_mega : bool
  ; is_reusable : bool
  ; is_verbose : bool
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
  ; tokenizer : Text_tokenizer.t
  ; bart_encoder : Dalle_bart_encoder.t option
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
  if result != 0
  then (
    let err = Sys_error (cmd ^ " command failed") in
    raise err)
  else ()
;;

let mkdir_with_path = "mkdir -p "

let check_and_create_dirs dalle_path vqgan_path =
  if Sys.file_exists dalle_path then () else exec_command @@ mkdir_with_path ^ dalle_path;
  if Sys.file_exists vqgan_path then () else exec_command @@ mkdir_with_path ^ vqgan_path
;;

let min_dalle_repo = "https://huggingface.co/kuprel/min-dalle/resolve/main/"
let image_token_count = 256

let fetch_tokenizer is_mega file_path =
  let suffix = if is_mega then "" else "_mini" in
  let parts = String.split_on_char '.' file_path in
  let name = List.hd parts in
  let name = List.hd @@ List.rev @@ String.split_on_char '/' name in
  let file_ext = List.hd @@ List.rev parts in
  let full_uri = min_dalle_repo ^ name ^ suffix ^ "." ^ file_ext in
  Stdio.printf "Tokenizer url:%s\n" full_uri;
  Client.get (Uri.of_string @@ full_uri)
  >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  if code != 200
  then Lwt.fail_with @@ file_path ^ " download failed"
  else
    Lwt_io.open_file file_path ~mode:Lwt_io.output
    >>= fun out_ch ->
    Cohttp_lwt.Body.write_body (fun body -> Lwt_io.write out_ch body) body
;;

let download_tokenizer is_verbose is_mega file_path =
  let is_downloaded = Sys.file_exists file_path in
  if not is_downloaded
  then (
    let _ = if is_verbose then print_string "downloading tokenizer params\n" else () in
    fetch_tokenizer is_mega file_path)
  else Lwt.return ()
;;

let load_file file_name =
  Lwt_io.open_file ~mode:Lwt_io.input file_name >>= fun in_ch -> Lwt_io.read in_ch
;;

let load_vocab vocab_path =
  load_file vocab_path
  >|= fun contents ->
  let out = Yojson.Safe.from_string contents in
  let out = Yojson.Safe.Util.to_assoc out in
  let out = List.map (fun (k, v) -> k, Yojson.Safe.Util.to_int v) out in
  let ht = Hashtbl.create (List.length out) in
  List.iter (fun (k, v) -> Hashtbl.add ht k v) out;
  ht
;;

let load_merges merges_path =
  load_file merges_path
  >|= fun contents ->
  let lines = String.split_on_char '\n' contents in
  List.tl lines |> List.filter (fun x -> Base.String.is_empty x == false)
;;

let init_tokenizer is_verbose is_mega vocab_path merges_path =
  let open Lwt.Syntax in
  let* resp, _body = Client.get (Uri.of_string @@ min_dalle_repo ^ "config.json") in
  let code = resp |> Response.status |> Code.code_of_status in
  if code != 200
  then Lwt.fail_with "HF config.json is not reachable"
  else
    let* _ = download_tokenizer is_verbose is_mega vocab_path in
    let* _ = download_tokenizer is_verbose is_mega merges_path in
    let tok_lwt = load_vocab vocab_path in
    let mrg_lwt = load_merges merges_path in
    let+ vocab, merges = Lwt.both tok_lwt mrg_lwt in
    Text_tokenizer.make vocab merges
;;

let fetch_encoder is_verbose is_mega encoder_path =
  let open Lwt.Syntax in
  let _ = if is_verbose then print_string "Downloading encoder\n" else () in
  let suffix = if is_mega then "" else "_mini" in
  let uri = Uri.of_string @@ min_dalle_repo ^ "encoder" ^ suffix ^ ".pt" in
  let* resp, body = Lwthttp.http_get_and_follow uri encoder_path in
  let code = resp |> Response.status |> Code.code_of_status in
  if code != 200
  then Lwt.fail_with @@ "HF Encoder is not reachable. Resp code:" ^ Int.to_string code
  else
    let* out_ch = Lwt_io.open_file encoder_path ~mode:Lwt_io.Output in
    Cohttp_lwt.Body.write_body (fun body -> Lwt_io.write out_ch body) body
;;

let download_encoder _frozen_vs is_verbose is_mega encoder_path =
  let is_downloaded = Sys.file_exists encoder_path in
  if is_downloaded then Lwt.return () else fetch_encoder is_verbose is_mega encoder_path
;;

let mk ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () : t Lwt.t =
  let open Lwt.Syntax in
  let is_mega = Option.value is_mega ~default:true in
  let is_verbose = Option.value is_verbose ~default:true in
  let is_reusable = Option.value is_reusable ~default:true in
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
  let device =
    Option.fold
      ~none:Torch_core.Device.Cpu
      ~some:(fun i -> Torch_core.Device.Cuda i)
      device
  in
  let vs = Torch.Var_store.create ~name:"dalle" ~device ~frozen:true () in
  let* tokenizer = init_tokenizer is_verbose is_mega vocab_path merges_path in
  let+ bart_encoder =
    if is_reusable
    then
      let+ _ = download_encoder vs is_verbose is_mega encoder_params_path in
      Option.some
      @@ Dalle_bart_encoder.make
           ~layer_count
           ~embed_count
           ~attention_head_count
           ~text_vocab_count
           ~text_token_count
           ~glu_embed_count
           ~vs
           ~device
    else Lwt.return Option.none
  in
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
  ; tokenizer
  ; bart_encoder
  }
;;

let make ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () =
  let open Lwt.Syntax in
  let+ m = mk ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () in
  print_string m.detoker_params_path;
  print_string m.decoder_params_path;
  print_string m.encoder_params_path;
  print_int m.image_vocab_count;
  print_int m.text_token_count;
  print_int m.glu_embed_count;
  print_int m.embed_count;
  print_int m.attention_head_count;
  print_int m.text_token_count;
  print_int m.layer_count;
  print_int m.text_vocab_count;
  print_int image_token_count;
  Printf.printf "%B\n" m.is_reusable;
  print_string m.models_root;
  (match Option.value m.dtype ~default:`f32 with
   | `f16 -> print_string "f16"
   | `f32 -> print_string "f32");
  let _ = Option.is_some m.bart_encoder in
  Printf.printf "%B\n" @@ Torch.Device.is_cuda m.device;
  List.iter print_string [ m.vocab_path; m.merges_path ];
  List.iter (Printf.printf "%B\n") [ m.is_reusable; m.is_verbose; m.is_mega ];
  m
;;

let generate_raw_image_stream
  ~text
  ~seed
  ~grid_size
  ?(is_seamless = false)
  ?(temperature = 1.)
  ?(top_k = 256)
  ?(supercondition_factor = 16)
  t
  =
  let open Torch in
  List.iter
    print_int
    [ top_k; supercondition_factor; seed; grid_size; Bool.to_int is_seamless ];
  List.iter print_float [ temperature ];
  let _image_count = Base.Int.pow grid_size 2 in
  if t.is_verbose then Stdio.printf "Tokenizing text..." else ();
  let tokens = Text_tokenizer.tokenize t.tokenizer ~text ~is_verbose:t.is_verbose in
  let tokens =
    if List.length tokens > t.text_token_count
    then Base.List.take tokens t.text_token_count
    else tokens
  in
  if t.is_verbose then Stdio.printf "%d text tokens" @@ List.length tokens;
  let text_tokens = Tensor.ones ~device:t.device [ 2; t.text_token_count ] ~kind:(Torch_core.Kind.(T i64)) in
  let tokens = Array.of_list tokens in
  Tensor.int_set text_tokens [ 0; 0 ] tokens.(0);
  Tensor.int_set text_tokens [ 0; 1 ] tokens.(Array.length tokens - 1);
  let indices =
    Tensor.range
      ~start:(Scalar.i 0)
      ~end_:(Scalar.i (Array.length tokens - 1))
      ~options:(Torch_core.Kind.(T i64), t.device)
  in
  let tokens = Bigarray.Array1.of_array Bigarray.Int Bigarray.C_layout tokens in
  let tokens = Tensor.of_bigarray @@ Bigarray.genarray_of_array1 tokens in
  let text_tokens =
    Tensor.index_put
      text_tokens
      ~indices:[ Some (Tensor.ones ~kind:(Torch_core.Kind.(T i64)) [1]); Some indices ]
      ~values:tokens
      ~accumulate:false
  in
  Tensor.print text_tokens;
  text_tokens
;;
