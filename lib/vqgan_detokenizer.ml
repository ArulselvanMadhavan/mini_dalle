open Torch

module GroupNorm = struct
  type t = { apply : Tensor.t -> Tensor.t }

  let make vs ~num_groups ~num_channels =
    let weight =
      Var_store.new_var
        vs
        ~trainable:false
        ~shape:[ num_channels ]
        ~init:Ones
        ~name:"weight"
    in
    let bias =
      Var_store.new_var
        vs
        ~trainable:false
        ~shape:[ num_channels ]
        ~init:Zeros
        ~name:"bias"
    in
    let apply xs =
      Tensor.group_norm
        ~num_groups
        ~weight:(Some weight)
        ~bias:(Some bias)
        ~eps:1e-5
        ~cudnn_enabled:false
        xs
    in
    { apply }
  ;;

  let forward t xs = t.apply xs
end

module ResnetBlock = struct
  type t =
    { conv1 : Nn.t
    ; conv2 : Nn.t
    ; nin_shortcut : Nn.t option
    ; norm1 : GroupNorm.t
    ; norm2 : GroupNorm.t
    }

  let make vs log2_count_in log2_count_out =
    let m, n = Base.Int.pow 2 log2_count_in, Base.Int.pow 2 log2_count_out in
    let is_middle = m == n in
    let norm1 =
      GroupNorm.make
        Var_store.(vs / "norm1")
        ~num_groups:(Base.Int.pow 2 5)
        ~num_channels:m
    in
    let norm2 =
      GroupNorm.make
        Var_store.(vs / "norm2")
        ~num_groups:(Base.Int.pow 2 5)
        ~num_channels:n
    in
    let conv1 =
      Layer.conv2d_ Var_store.(vs / "conv1") ~ksize:3 ~stride:1 ~padding:1 ~input_dim:m n
    in
    let conv2 =
      Layer.conv2d_ Var_store.(vs / "conv2") ~ksize:3 ~stride:1 ~padding:1 ~input_dim:n n
    in
    let nin_shortcut =
      if not is_middle
      then
        Some
          (Layer.conv2d_
             Var_store.(vs / "nin_shortcut")
             ~ksize:1
             ~stride:1
             ~input_dim:m
             n)
      else None
    in
    { conv1; conv2; nin_shortcut; norm1; norm2 }
  ;;

  let forward t x =
    let h = x in
    let h = GroupNorm.forward t.norm1 h in
    let h = Tensor.(h * sigmoid h) in
    let h = Layer.forward t.conv1 h in
    let h = GroupNorm.forward t.norm2 h in
    let h = Tensor.(h * sigmoid h) in
    let h = Layer.forward t.conv2 h in
    let x = Option.fold ~some:(fun l -> Layer.forward l x) ~none:x t.nin_shortcut in
    Tensor.(x + h)
  ;;
end

module AttentionBlock = struct
  type t =
    { q : Nn.t
    ; k : Nn.t
    ; v : Nn.t
    ; proj_out : Nn.t
    ; norm : GroupNorm.t
    }

  let make vs =
    let n = Base.Int.pow 2 9 in
    let q = Layer.conv2d_ Var_store.(vs / "q") ~ksize:1 ~stride:1 ~input_dim:n n in
    let k = Layer.conv2d_ Var_store.(vs / "k") ~ksize:1 ~stride:1 ~input_dim:n n in
    let v = Layer.conv2d_ Var_store.(vs / "v") ~ksize:1 ~stride:1 ~input_dim:n n in
    let proj_out =
      Layer.conv2d_ Var_store.(vs / "proj_out") ~ksize:1 ~stride:1 ~input_dim:n n
    in
    let norm =
      GroupNorm.make
        Var_store.(vs / "norm")
        ~num_groups:(Base.Int.pow 2 5)
        ~num_channels:n
    in
    { q; k; v; proj_out; norm }
  ;;

  let forward t x =
    let n, m = Base.Int.pow 2 9, List.hd (Tensor.shape x) in
    let h = x in
    let h = GroupNorm.forward t.norm h in
    let k = Layer.forward t.k h in
    let v = Layer.forward t.v h in
    let q = Layer.forward t.q h in
    let k = Tensor.reshape k ~shape:[ m; n; -1 ] in
    let v = Tensor.reshape v ~shape:[ m; n; -1 ] in
    let q = Tensor.reshape q ~shape:[ m; n; -1 ] in
    let q = Tensor.permute q ~dims:[ 0; 2; 1 ] in
    let w = Tensor.bmm q ~mat2:k in
    let w = Tensor.(div_scalar w (Scalar.f (Base.Float.sqrt (Float.of_int n)))) in
    let w = Tensor.softmax w ~dim:2 ~dtype:(T Float) in
    let w = Tensor.permute w ~dims:[ 0; 2; 1 ] in
    let h = Tensor.bmm v ~mat2:w in
    let token_count = Base.List.last_exn (Tensor.shape h) in
    let token_count = Int.of_float (Float.sqrt (Float.of_int token_count)) in
    let h = Tensor.reshape h ~shape:[ m; n; token_count; token_count ] in
    let h = Layer.forward t.proj_out h in
    Tensor.(x + h)
  ;;
end

module MiddleLayer = struct
  type t =
    { block_1 : ResnetBlock.t
    ; attn_1 : AttentionBlock.t
    ; block_2 : ResnetBlock.t
    }

  let make vs =
    let block_1 = ResnetBlock.make Var_store.(vs / "block_1") 9 9 in
    let attn_1 = AttentionBlock.make Var_store.(vs / "attn_1") in
    let block_2 = ResnetBlock.make Var_store.(vs / "block_2") 9 9 in
    { block_1; attn_1; block_2 }
  ;;

  let forward t h =
    let h = ResnetBlock.forward t.block_1 h in
    let h = AttentionBlock.forward t.attn_1 h in
    let h = ResnetBlock.forward t.block_2 h in
    h
  ;;
end

module Upsample = struct
  type t = { conv : Nn.t }

  let make vs log2_count =
    let n = Base.Int.pow 2 log2_count in
    let conv =
      Layer.conv2d_ Var_store.(vs / "conv") ~ksize:3 ~stride:1 ~padding:1 ~input_dim:n n
    in
    { conv }
  ;;

  let forward t x =
    let _, _, h, w = Tensor.shape4_exn x in
    let x =
      Tensor.upsample_nearest2d
        ~output_size:[ 2 * h; 2 * w ]
        ~scales_h:None
        ~scales_w:None
        x
    in
    Layer.forward t.conv x
  ;;
end

module UpsampleBlock = struct
  type t =
    { attn : AttentionBlock.t array option
    ; block : ResnetBlock.t array
    ; upsample : Upsample.t option
    }

  let make vs ~log2_count_in ~log2_count_out ~has_attention ~has_upsample =
    let block =
      [| ResnetBlock.make Var_store.(vs / "block" // 0) log2_count_in log2_count_out
       ; ResnetBlock.make Var_store.(vs / "block" // 1) log2_count_out log2_count_out
       ; ResnetBlock.make Var_store.(vs / "block" // 2) log2_count_out log2_count_out
      |]
    in
    let attn =
      if has_attention
      then
        Some
          [| AttentionBlock.make Var_store.(vs / "attn" // 0)
           ; AttentionBlock.make Var_store.(vs / "attn" // 1)
           ; AttentionBlock.make Var_store.(vs / "attn" // 2)
          |]
      else None
    in
    let upsample =
      if has_upsample
      then Some (Upsample.make Var_store.(vs / "upsample") log2_count_out)
      else None
    in
    { attn; block; upsample }
  ;;

  let forward t h =
    let h = ref h in
    for i = 0 to Array.length t.block - 1 do
      h := ResnetBlock.forward t.block.(i) !h;
      h := Option.fold t.attn ~none:!h ~some:(fun xs -> AttentionBlock.forward xs.(i) !h)
    done;
    let h = !h in
    Option.fold t.upsample ~none:h ~some:(fun l -> Upsample.forward l h)
  ;;
end

module Decoder = struct
  type t =
    { conv_in : Nn.t
    ; conv_out : Nn.t
    ; mid : MiddleLayer.t
    ; up : UpsampleBlock.t array
    ; norm_out : GroupNorm.t
    }

  let make vs =
    let conv_in =
      Layer.conv2d_
        Var_store.(vs / "conv_in")
        ~ksize:3
        ~stride:1
        ~padding:1
        ~input_dim:(Base.Int.pow 2 8)
        (Base.Int.pow 2 9)
    in
    let mid = MiddleLayer.make Var_store.(vs / "mid") in
    let up =
      [| UpsampleBlock.make
           Var_store.(vs / "up" // 0)
           ~log2_count_in:7
           ~log2_count_out:7
           ~has_attention:false
           ~has_upsample:false
       ; UpsampleBlock.make
           Var_store.(vs / "up" // 1)
           ~log2_count_in:8
           ~log2_count_out:7
           ~has_attention:false
           ~has_upsample:true
       ; UpsampleBlock.make
           Var_store.(vs / "up" // 2)
           ~log2_count_in:8
           ~log2_count_out:8
           ~has_attention:false
           ~has_upsample:true
       ; UpsampleBlock.make
           Var_store.(vs / "up" // 3)
           ~log2_count_in:9
           ~log2_count_out:8
           ~has_attention:false
           ~has_upsample:true
       ; UpsampleBlock.make
           Var_store.(vs / "up" // 4)
           ~log2_count_in:9
           ~log2_count_out:9
           ~has_attention:true
           ~has_upsample:true
      |]
    in
    let norm_out =
      GroupNorm.make
        Var_store.(vs / "norm_out")
        ~num_groups:(Base.Int.pow 2 5)
        ~num_channels:(Base.Int.pow 2 7)
    in
    let conv_out =
      Layer.conv2d_
        Var_store.(vs / "conv_out")
        ~ksize:3
        ~stride:1
        ~padding:1
        ~input_dim:(Base.Int.pow 2 7)
        3
    in
    { conv_in; conv_out; mid; up; norm_out }
  ;;

  let forward t z =
    let z = Layer.forward t.conv_in z in
    let z = MiddleLayer.forward t.mid z in
    let z = ref z in
    let max_idx = Array.length t.up - 1 in
    for idx = 0 to max_idx do
      let idx = max_idx - idx in
      z := UpsampleBlock.forward t.up.(idx) !z
    done;
    let z = !z in
    let z = GroupNorm.forward t.norm_out z in
    let z = Tensor.(z * sigmoid z) in
    Layer.forward t.conv_out z
  ;;
end

(* let print_named_tensors xs = *)
(*   List.iteri (fun i (name, t) -> Stdio.printf "%d)%s|%s\n" i name (Tensor.shape_str t)) xs *)

(* VQGAN *)
type t =
  { embedding : Nn.t
  ; post_quant_conv : Nn.t
  ; decoder : Decoder.t
  }

let make vs =
  let vocab_count = Base.Int.pow 2 14 in
  let embed_count = Base.Int.pow 2 8 in
  let embedding =
    Layer.embeddings
      Var_store.(vs / "embedding")
      ~num_embeddings:vocab_count
      ~embedding_dim:embed_count
  in
  let post_quant_conv =
    Layer.conv2d_
      Var_store.(vs / "post_quant_conv")
      ~ksize:1
      ~stride:1
      ~input_dim:embed_count
      embed_count
  in
  let decoder = Decoder.make Var_store.(vs / "decoder") in
  (* print_named_tensors (Var_store.all_vars vs); *)
  Serialize.load_multi_
    ~named_tensors:(Var_store.all_vars vs)
    ~filename:"extracts/detokermega/detoker.ot";
  Stdio.printf "****Detoker complete*****\n";
  { embedding; post_quant_conv; decoder }
;;

let forward t ~is_seamless z =
  let grid_size = Int.of_float (Float.sqrt (Float.of_int (List.hd (Tensor.shape z)))) in
  let token_count = grid_size * Base.Int.pow 2 4 in
  let z =
    if is_seamless
    then (
      let z =
        Tensor.view z ~size:[ grid_size; grid_size; Base.Int.pow 2 4; Base.Int.pow 2 4 ]
      in
      let z_shp = Array.of_list (Tensor.shape z) in
      let z = Tensor.reshape z ~shape:[ z_shp.(0); z_shp.(1) * z_shp.(2); z_shp.(3) ] in
      let z = Tensor.transpose z ~dim0:1 ~dim1:0 in
      let z_shp = Array.of_list (Tensor.shape z) in
      let z = Tensor.reshape z ~shape:[ z_shp.(0); z_shp.(1) * z_shp.(2) ] in
      let z = Tensor.reshape z ~shape:[ -1; 1 ] in
      let z = Layer.forward t.embedding z in
      Tensor.view z ~size:[ 1; token_count; token_count; Base.Int.pow 2 8 ])
    else (
      let z = Layer.forward t.embedding z in
      Tensor.view
        z
        ~size:
          [ List.hd (Tensor.shape z)
          ; Base.Int.pow 2 4
          ; Base.Int.pow 2 4
          ; Base.Int.pow 2 8
          ])
  in
  let z = Tensor.permute z ~dims:[ 0; 3; 1; 2 ] in
  let z = Tensor.contiguous z in
  let z = Layer.forward t.post_quant_conv z in
  let z = Decoder.forward t.decoder z in
  let z = Tensor.permute z ~dims:[ 0; 2; 3; 1 ] in
  let z = Tensor.clip z ~min:(Scalar.f 0.) ~max:(Scalar.f 1.) in
  let z = Tensor.mul_scalar z (Scalar.i 255) in
  if is_seamless
  then Tensor.index z ~indices:[ Some (Tensor.of_int0 0) ]
  else (
    let z =
      Tensor.view z ~size:[ grid_size; grid_size; Base.Int.pow 2 8; Base.Int.pow 2 8; 3 ]
    in
    let z_shp = Array.of_list (Tensor.shape z) in
    let z =
      Tensor.reshape z ~shape:[ z_shp.(0); z_shp.(1) * z_shp.(2); z_shp.(3); z_shp.(4) ]
    in
    let z = Tensor.transpose z ~dim0:1 ~dim1:0 in
    let z_shp = Array.of_list (Tensor.shape z) in
    Tensor.reshape z ~shape:[ z_shp.(0); z_shp.(1) * z_shp.(2); z_shp.(3) ])
;;
