open Mini_dalle

let exec_md text output_file device =
  let open Lwt.Syntax in
  let+ m = Min_dalle.make ?device () in
  Min_dalle.generate_raw_image_stream ~text ~grid_size:3 ~output_file ~seed:(-1) m
;;

let run_md text output_file device =
  let start = Unix.gettimeofday () in
  let m = exec_md text output_file device in
  let _ = Lwt_main.run m in
  let stop = Unix.gettimeofday () in
  Stdio.printf "Done.%f\n" (stop -. start);
  Stdio.Out_channel.flush stdout
;;

let () =
  let open Cmdliner in
  let text =
    Arg.(
      required
      & pos 0 (some string) None
      & info [] ~docv:"TEXT" ~doc:"Text to turn into image")
  in
  let fname =
    Arg.(
      required
      & pos 1 (some string) None
      & info [] ~docv:"FILENAME" ~doc:"Output Filename")
  in
  let device =
    Arg.(value & opt (some int) None & info [ "device" ] ~docv:"DEVICE" ~doc:"Device Id")
  in
  let doc = "Mini dalle - Text to Image generation" in
  let man = [ `S "DESCRIPTION"; `P "Turn text into image" ] in
  let cmd =
    Term.(const run_md $ text $ fname $ device), Cmd.info "mini-dalle" ~sdocs:"" ~doc ~man
  in
  let default_cmd = Term.(ret (const (`Help (`Pager, None)))) in
  let info =
    let doc = "Mini_dalle: Text to Image generation" in
    Cmd.info "mini_dalle" ~version:"0.0.1" ~sdocs:"" ~doc
  in
  let cmds = [ cmd ] |> List.map (fun (cmd, info) -> Cmd.v info cmd) in
  let main_cmd = Cmd.group info ~default:default_cmd cmds in
  Cmd.eval main_cmd |> Caml.exit
;;
(* (\* let m = Min_dalle.fetch_file "encoder" true true "pretrained/dalle_bart_mega/encoder.pt" in *\) *)
(* let m = *)

(* in *)

(* (\* let _ = *\) *)
(* (\*   Min_dalle.generate_raw_image_stream *\) *)
(* (\*     ~text:"cactus in a corn field" *\) *)
(* (\*     ~seed:42 *\) *)
(* (\*     ~grid_size:3 *\) *)
(* (\*     m *\) *)
(* (\* in *\) *)
(* (\* let _ = Min_dalle.image_grid_from_tokens m in *\) *)
