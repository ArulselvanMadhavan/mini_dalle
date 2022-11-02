open Mini_dalle

let () =
  let m = Min_dalle.make () in
  let m = Lwt_main.run m in
  let _ =
    Min_dalle.generate_raw_image_stream
      ~text:"cactus in corn field"
      ~seed:(-1)
      ~grid_size:3
      m
  in
  print_string "done";
  Stdio.Out_channel.flush stdout
;;
