open Cohttp_lwt_unix
open Cohttp
open Lwt

type t =
  { dtype : [ `f32 | `f16 ] option
  ; device : Torch_core.Device.t
  ; is_mega : bool
  ; is_reusable : bool
  ; is_verbose : bool
  ; layer_count : int
  ; text_token_count : int
  ; embed_count : int
  ; tokenizer : Text_tokenizer.t (* Mutable? *)
  ; mutable bart_encoder : Dalle_bart_encoder.t option
  ; mutable bart_decoder : Dalle_bart_decoder.t option
  ; mutable detokenizer : Vqgan_detokenizer.t option
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

let rename_pt_to_ot pt_path =
  let ot_path = Base.String.drop_suffix pt_path 2 in
  ot_path ^ "ot"
;;

let generate_ot file_type pt_path =
  let extracts_dir = "extracts/" ^ file_type ^ "mega/" in
  let filename = file_type ^ ".ot" in
  Serialize.serialize_model extracts_dir filename;
  let ot_path = rename_pt_to_ot pt_path in
  let cmd = "mv " ^ extracts_dir ^ filename ^ " " ^ ot_path in
  let _ = Sys.command cmd in
  let cmd = "rm -rf " ^ extracts_dir in
  let _ = Sys.command cmd in
  let cmd = "rm -rf " ^ pt_path in
  let _ = Sys.command cmd in
  ()
;;

let fetch_file file_type is_verbose is_mega pt_path =
  let open Lwt.Syntax in
  let _ = if is_verbose then print_string ("Downloading " ^ file_type ^ "\n") else () in
  let suffix = if is_mega then "" else "_mini" in
  let uri = Uri.of_string @@ min_dalle_repo ^ file_type ^ suffix ^ ".pt" in
  let* resp, _ = Lwthttp.http_get_and_follow uri pt_path in
  let code = resp |> Response.status |> Code.code_of_status in
  if code != 200
  then
    Lwt.fail_with
    @@ "HF"
    ^ file_type
    ^ " is not reachable. Resp code:"
    ^ Int.to_string code
  else (
    let cmd = "python3 scripts/export_" ^ file_type ^ ".py" in
    let result = Sys.command cmd in
    if result == 0
    then Lwt.return (generate_ot file_type pt_path)
    else Lwt.fail_with (file_type ^ " extract pt to npy failed"))
;;

let download_file file_type is_verbose is_mega pt_path =
  let is_downloaded = Sys.file_exists (rename_pt_to_ot pt_path) in
  if is_downloaded then Lwt.return () else fetch_file file_type is_verbose is_mega pt_path
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
  let* tokenizer = init_tokenizer is_verbose is_mega vocab_path merges_path in
  let encoder_lwt = download_file "encoder" is_verbose is_mega encoder_params_path in
  let decoder_lwt = download_file "decoder" is_verbose is_mega decoder_params_path in
  let detoker_lwt = download_file "detoker" is_verbose is_mega detoker_params_path in
  let+ _ = Lwt.all [ encoder_lwt; decoder_lwt; detoker_lwt ] in
  let vs = Torch.Var_store.create ~name:"encoder" ~device ~frozen:true () in
  let bart_encoder =
    Some
      (Dalle_bart_encoder.make
         ~layer_count
         ~embed_count
         ~attention_head_count
         ~text_vocab_count
         ~text_token_count
         ~glu_embed_count
         ~vs
         ~device
         ~params_path:(rename_pt_to_ot encoder_params_path))
  in
  let vs = Torch.Var_store.create ~name:"decoder" ~device ~frozen:true () in
  let bart_decoder =
    Some
      (Dalle_bart_decoder.make
         vs
         ~image_vocab_count
         ~embed_count
         ~attention_head_count
         ~glu_embed_count
         ~layer_count
         ~device
         ~params_path:(rename_pt_to_ot decoder_params_path))
  in
  let vs = Torch.Var_store.create ~name:"detoker" ~device ~frozen:true () in
  let detokenizer =
    Some (Vqgan_detokenizer.make vs ~params_path:(rename_pt_to_ot detoker_params_path))
  in
  { dtype
  ; device
  ; is_mega
  ; is_reusable
  ; is_verbose
  ; layer_count
  ; text_token_count
  ; embed_count
  ; tokenizer
  ; bart_encoder
  ; bart_decoder
  ; detokenizer
  }
;;

let make ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () =
  let open Lwt.Syntax in
  let+ m = mk ?models_root ?dtype ?device ?is_mega ?is_reusable ?is_verbose () in
  (match Option.value m.dtype ~default:`f32 with
   | `f16 -> print_string "f16"
   | `f32 -> print_string "f32");
  List.iter (Printf.printf "%B\n") [ m.is_reusable; m.is_verbose; m.is_mega ];
  m
;;

let generate_raw_image_stream
  ~text
  ~seed
  ~grid_size
  ~output_file
  ?(is_seamless = false)
  ?(temperature = 1.)
  ?(top_k = 256)
  ?(supercondition_factor = 16)
  t
  =
  let open Torch in
  List.iter print_int [ top_k; seed; supercondition_factor; Bool.to_int is_seamless ];
  List.iter print_float [ temperature ];
  let image_count = Base.Int.pow grid_size 2 in
  if t.is_verbose then Stdio.printf "Tokenizing text...\n" else ();
  let tokens = Text_tokenizer.tokenize t.tokenizer ~text ~is_verbose:t.is_verbose in
  let tokens =
    if List.length tokens > t.text_token_count
    then Base.List.take tokens t.text_token_count
    else tokens
  in
  if t.is_verbose then Stdio.printf "%d text tokens\n" @@ List.length tokens;
  let text_tokens =
    Tensor.ones ~device:t.device [ 2; t.text_token_count ] ~kind:Torch_core.Kind.(T i64)
  in
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
  let tokens =
    Tensor.of_bigarray ~device:t.device @@ Bigarray.genarray_of_array1 tokens
  in
  let text_tokens =
    Tensor.index_put
      text_tokens
      ~indices:[ Some (Tensor.ones ~kind:Torch_core.Kind.(T i64) [ 1 ]); Some indices ]
      ~values:tokens
      ~accumulate:false
  in
  if t.is_verbose then Stdio.printf "Encoding text tokens\n" else ();
  let encoder_state =
    Dalle_bart_encoder.forward (Option.get t.bart_encoder) ~text_tokens
  in
  t.bart_encoder <- None;
  Caml.Gc.full_major ();
  Stdio.printf "Finished with Encoder\n";
  Stdio.Out_channel.flush stdout;
  (* Torch.Serialize.save encoder_state ~filename:"encoder_state.ot"; *)
  let expanded_indices =
    Tensor.concat
      [ Tensor.repeat (Tensor.of_int0 0) ~repeats:[ image_count ]
      ; Tensor.repeat (Tensor.of_int0 1) ~repeats:[ image_count ]
      ]
      ~dim:0
  in
  let encoder_state = Tensor.index encoder_state ~indices:[ Some expanded_indices ] in
  let text_tokens = Tensor.index text_tokens ~indices:[ Some expanded_indices ] in
  let attention_mask = Tensor.not_equal text_tokens (Scalar.i 1) in
  let mask_shp = Tensor.shape attention_mask in
  let attention_mask =
    Tensor.reshape
      attention_mask
      ~shape:[ List.hd mask_shp; 1; 1; Base.List.last_exn mask_shp ]
  in
  let attention_state =
    Tensor.zeros
      ~requires_grad:false
      ~device:t.device
      [ t.layer_count; image_count * 4; Constants.image_token_count; t.embed_count ]
  in
  let image_tokens =
    Tensor.full
      ~size:[ image_count; Constants.image_token_count + 1 ]
      ~fill_value:(Scalar.i (Base.Int.pow 2 14 - 1))
      ~options:(Torch_core.Kind.(T Int64), t.device)
  in
  if seed > 0 then Torch_core.Wrapper.manual_seed seed else ();
  let token_indices =
    Tensor.arange
      ~end_:(Scalar.i Constants.image_token_count)
      ~options:(Torch_core.Kind.(T Int64), t.device)
  in
  let settings =
    Tensor.of_float1
      [| temperature; Float.of_int top_k; Float.of_int supercondition_factor |]
      ~device:t.device
  in
  let attention_state = ref attention_state in
  let image_tokens = ref image_tokens in
  for i = 0 to Constants.image_token_count - 1 do
    Caml.Gc.full_major ();
    let prev_tokens =
      Tensor.index !image_tokens ~indices:[ None; Some (Tensor.of_int0 i) ]
    in
    let prev_tokens = Tensor.unsqueeze prev_tokens ~dim:1 in
    let token_index = Tensor.index token_indices ~indices:[ Some (Tensor.of_int0 i) ] in
    let image_token, attention_state_0 =
      Dalle_bart_decoder.sample_tokens
        (Option.get t.bart_decoder)
        ~settings
        ~attention_mask
        ~encoder_state
        ~attention_state:!attention_state
        ~prev_tokens
        ~token_index
    in
    let image_token = Tensor.squeeze image_token in
    image_tokens
      := Tensor.index_put_
           !image_tokens
           ~indices:[ None; Some (Tensor.of_int0 (i + 1)) ]
           ~values:image_token
           ~accumulate:false;
    attention_state := attention_state_0
  done;
  Stdio.printf "Finished with Decoder\n";
  Stdio.Out_channel.flush stdout;
  t.bart_decoder <- None;
  Caml.Gc.full_major ();
  (* Gen images *)
  let image_tokens =
    Tensor.slice !image_tokens ~dim:1 ~start:(Some 1) ~end_:None ~step:1
  in
  let images =
    Vqgan_detokenizer.forward (Option.get t.detokenizer) ~is_seamless image_tokens
  in
  let images = Tensor.to_dtype images ~dtype:(T Uint8) ~non_blocking:true ~copy:false in
  Torch_vision.Image.write_image images ~filename:output_file;
  t.detokenizer <- None;
  Caml.Gc.full_major ();
  Stdio.printf "Finished with Detoker\n";
  Stdio.Out_channel.flush stdout;
  images
;;
