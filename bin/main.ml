open Mini_dalle

let () =
  let m = Min_dalle.make () in
  let _m = Lwt_main.run m in
  print_string "done";
  Stdio.Out_channel.flush stdout;
;;
