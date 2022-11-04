open Torch

module ResnetBlock = struct
  type t =
    { m : int
    ; n : int
    ; conv1 : Nn.t
    ; conv2 : Nn.t
    ; nin_shortcut : Nn.t option
    }

  let make vs log2_count_in log2_count_out =
    let m, n = Base.Int.pow 2 log2_count_in, Base.Int.pow 2 log2_count_out in
    let is_middle = m == n in
    let conv1 =
      Layer.conv2d
        Var_store.(vs / "conv1")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:m
        n
    in
    let conv2 =
      Layer.conv2d
        Var_store.(vs / "conv2")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:n
        n
    in
    let nin_shortcut =
      if not is_middle
      then
        Some
          (Layer.conv2d
             Var_store.(vs / "nin_shortcut")
             ~ksize:(1, 1)
             ~stride:(1, 1)
             ~input_dim:m
             n)
      else None
    in
    { m; n; conv1; conv2; nin_shortcut }
  ;;

  let forward t x =
    let h = x in
    (* FIXME *)
    let h =
      Tensor.group_norm
        h
        ~num_groups:(Base.Int.pow 2 5)
        ~cudnn_enabled:false
        ~eps:1e-5
        ~weight:None
        ~bias:None
    in
    let h = Tensor.(h * sigmoid h) in
    let h = Layer.forward t.conv1 h in
    (* FIXME *)
    let h =
      Tensor.group_norm
        h
        ~num_groups:(Base.Int.pow 2 5)
        ~cudnn_enabled:false
        ~eps:1e-5
        ~weight:None
        ~bias:None
    in
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
    ; n : int
    }

  let make vs =
    let n = Base.Int.pow 2 9 in
    let q =
      Layer.conv2d Var_store.(vs / "q") ~ksize:(1, 1) ~stride:(1, 1) ~input_dim:n n
    in
    let k =
      Layer.conv2d Var_store.(vs / "k") ~ksize:(1, 1) ~stride:(1, 1) ~input_dim:n n
    in
    let v =
      Layer.conv2d Var_store.(vs / "v") ~ksize:(1, 1) ~stride:(1, 1) ~input_dim:n n
    in
    let proj_out =
      Layer.conv2d Var_store.(vs / "proj_out") ~ksize:(1, 1) ~stride:(1, 1) ~input_dim:n n
    in
    { q; k; v; proj_out; n }
  ;;

  let forward t x =
    let n, m = Base.Int.pow 2 9, List.hd (Tensor.shape x) in
    let h = x in
    let h =
      Tensor.group_norm
        h
        ~num_groups:t.n
        ~eps:1e-5
        ~cudnn_enabled:false
        ~weight:None
        ~bias:None
    in
    let k = Layer.forward t.k h in
    let v = Layer.forward t.v h in
    let q = Layer.forward t.q h in
    let k = Tensor.reshape k ~shape:[ m; n; -1 ] in
    let v = Tensor.reshape v ~shape:[ m; n; -1 ] in
    let q = Tensor.reshape q ~shape:[ m; n; -1 ] in
    let q = Tensor.permute q ~dims:[ 0; 2; 1 ] in
    let w = Tensor.bmm q ~mat2:k in
    let w =
      Tensor.(
        div_scalar w (Scalar.i (Base.Int.of_float (Base.Float.sqrt (Float.of_int n)))))
    in
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
    let block_1 = ResnetBlock.make vs 9 9 in
    let attn_1 = AttentionBlock.make vs in
    let block_2 = ResnetBlock.make vs 9 9 in
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
      Layer.conv2d
        Var_store.(vs / "conv")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:n
        n
    in
    { conv }
  ;;

  let forward t x =
    let output_size = Base.List.take (Base.List.rev (Tensor.shape x)) 2 in
    let output_size = Base.List.map ~f:(Int.mul 2) output_size in
    let x = Tensor.upsample_nearest2d x ~output_size ~scales_h:None ~scales_w:None in
    Layer.forward t.conv x
  ;;
end

module UpsampleBlock = struct
  type t =
    { has_attention : bool
    ; has_upsample : bool
    ; attn : AttentionBlock.t array option
    ; block : ResnetBlock.t list
    ; upsample : Upsample.t option
    }

  let make vs ~log2_count_in ~log2_count_out ~has_attention ~has_upsample =
    let block =
      [ ResnetBlock.make vs log2_count_in log2_count_out
      ; ResnetBlock.make vs log2_count_out log2_count_out
      ; ResnetBlock.make vs log2_count_out log2_count_out
      ]
    in
    let attn =
      if has_attention
      then
        Some [| AttentionBlock.make vs; AttentionBlock.make vs; AttentionBlock.make vs |]
      else None
    in
    let upsample =
      if has_upsample then Some (Upsample.make vs log2_count_out) else None
    in
    { has_attention; has_upsample; attn; block; upsample }
  ;;

  let forward t h =
    Base.List.foldi
      ~f:(fun i acc bl ->
        let h = ResnetBlock.forward bl acc in
        let h =
          Option.fold t.attn ~none:h ~some:(fun abl -> AttentionBlock.forward abl.(i) h)
        in
        Option.fold t.upsample ~none:h ~some:(fun us -> Upsample.forward us h))
      ~init:h
      t.block
  ;;
end

module Decoder = struct
  type t =
    { conv_in : Nn.t
    ; conv_out : Nn.t
    ; mid : MiddleLayer.t
    ; up : UpsampleBlock.t list
    }

  let make vs =
    let conv_in =
      Layer.conv2d
        Var_store.(vs / "conv_in")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:(Base.Int.pow 2 8)
        (Base.Int.pow 2 9)
    in
    let mid = MiddleLayer.make vs in
    let up =
      [ UpsampleBlock.make
          vs
          ~log2_count_in:7
          ~log2_count_out:7
          ~has_attention:false
          ~has_upsample:false
      ; UpsampleBlock.make
          vs
          ~log2_count_in:8
          ~log2_count_out:7
          ~has_attention:false
          ~has_upsample:true
      ; UpsampleBlock.make
          vs
          ~log2_count_in:8
          ~log2_count_out:8
          ~has_attention:false
          ~has_upsample:true
      ; UpsampleBlock.make
          vs
          ~log2_count_in:9
          ~log2_count_out:8
          ~has_attention:false
          ~has_upsample:true
      ; UpsampleBlock.make
          vs
          ~log2_count_in:9
          ~log2_count_out:9
          ~has_attention:true
          ~has_upsample:true
      ]
    in
    let conv_out =
      Layer.conv2d
        Var_store.(vs / "conv_out")
        ~ksize:(3, 3)
        ~stride:(1, 1)
        ~padding:(1, 1)
        ~input_dim:(Base.Int.pow 2 7)
        3
    in
    { conv_in; conv_out; mid; up }
  ;;

  let forward t z =
    let z = Layer.forward t.conv_in z in
    let z = MiddleLayer.forward t.mid z in
    let z =
      Base.List.fold_left
        (Base.List.rev t.up)
        ~init:z
        ~f:(Base.Fn.flip UpsampleBlock.forward)
    in
    let z =
      Tensor.group_norm
        z
        ~num_groups:(Base.Int.pow 2 5)
        ~eps:1e-5
        ~cudnn_enabled:false
        ~weight:None
        ~bias:None
    in
    let z = Tensor.(z * sigmoid z) in
    Layer.forward t.conv_out z
  ;;
end

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
    Layer.conv2d
      Var_store.(vs / "post_quant_conv")
      ~ksize:(1, 1)
      ~stride:(1, 1)
      ~input_dim:embed_count
      embed_count
  in
  let decoder = Decoder.make Var_store.(vs / "decoder") in
  { embedding; post_quant_conv; decoder }
;;

let forward t is_seamless z =
  let grid_size = Int.of_float (Float.sqrt (Float.of_int (List.hd (Tensor.shape z)))) in
  let token_count = Base.Int.pow (grid_size * 2) 4 in
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
