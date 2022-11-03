open Mini_dalle

let () =
  let start = Unix.gettimeofday () in
  let m = Min_dalle.make ~device:0 () in
  let m = Lwt_main.run m in
  let _ =
    Min_dalle.generate_raw_image_stream
      ~text:"cactus in a corn field"
      ~seed:(-1)
      ~grid_size:3
      m
  in
  let stop = Unix.gettimeofday () in
  Stdio.printf "Done.%f\n" (stop -. start);
  Stdio.Out_channel.flush stdout
;;
