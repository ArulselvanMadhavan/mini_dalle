open Torch
open Base

let learning_rate = Tensor.f 1.

let () =
  let { Dataset_helper.train_images; train_labels; test_images; test_labels } =
    Mnist_helper.read_files ()
  in
  let test_count = Tensor.shape test_images |> List.hd_exn |> Float.of_int in
  Stdio.printf "%s\n" @@ Tensor.shape_str train_images;
  Stdio.printf "%s\n" @@ Tensor.shape_str train_labels;
  Stdio.printf "%s\n" @@ Tensor.shape_str test_images;
  Stdio.printf "%s\n" @@ Tensor.shape_str test_labels;
  let ws = Tensor.zeros Mnist_helper.[ image_dim; label_count ] ~requires_grad:true in
  let bs = Tensor.zeros [ Mnist_helper.label_count ] ~requires_grad:true in
  let model xs = Tensor.(mm xs ws + bs) in
  for index = 1 to 200 do
    let loss =
      Tensor.cross_entropy_for_logits (model train_images) ~targets:train_labels
    in
    Tensor.backward loss;
    (* Grad descent *)
    Tensor.(
      no_grad (fun () ->
        ws -= (grad ws * learning_rate);
        bs -= (grad bs * learning_rate)));
    Tensor.zero_grad ws;
    Tensor.zero_grad bs;
    (* Compute the validation error. *)
    let test_accuracy =
      Tensor.(argmax ~dim:(-1) (model test_images) = test_labels)
      |> Tensor.to_kind ~kind:(T Float)
      |> Tensor.sum
      |> Tensor.float_value
      |> fun sum -> sum /. test_count
    in
    Stdio.printf "%d %f %.2f%%\n%!" index (Tensor.float_value loss) (100. *. test_accuracy);
    Caml.Gc.full_major ()
  done
;;
