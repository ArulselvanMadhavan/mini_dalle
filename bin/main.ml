open Mini_dalle

let () =
  let start = Unix.gettimeofday () in
  (* let m = Min_dalle.make ~device:0 () in *)
  (* let m = Min_dalle.fetch_file "encoder" true true "pretrained/dalle_bart_mega/encoder.pt" in *)
  let m =
    Min_dalle.fetch_file "detoker" true true "pretrained/dalle_bart_mega/detoker.pt"
  in
  let _ = Lwt_main.run m in
  (* let _ = *)
  (*   Min_dalle.generate_raw_image_stream *)
  (*     ~text:"cactus in a corn field" *)
  (*     ~seed:42 *)
  (*     ~grid_size:3 *)
  (*     m *)
  (* in *)
  (* let _ = Min_dalle.image_grid_from_tokens m in *)
  let stop = Unix.gettimeofday () in
  Stdio.printf "Done.%f\n" (stop -. start);
  Stdio.Out_channel.flush stdout
;;
