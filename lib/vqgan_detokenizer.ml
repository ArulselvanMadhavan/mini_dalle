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
  type t

  let make vs =
    let n = Base.Int.pow 2 9 in
    (* let norm =  *)
end


module MiddleLayer = struct
  type t = { block_1 : ResnetBlock.t }

  let make vs =
    let block_1 = ResnetBlock.make vs 9 9 in
    { block_1 }
  ;;
end

module Decoder = struct
  type t = { conv_in : Nn.t }

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
    { conv_in }
  ;;
end

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
  { embedding; post_quant_conv }
;;
