let rec http_get_and_follow uri enc_path =
  let open Lwt.Syntax in
  let* ans = Cohttp_lwt_unix.Client.get uri in
  follow_redirect ~enc_path ans

and follow_redirect ~enc_path (response, body) =
  let open Lwt.Syntax in
  let status = Cohttp.Response.status response in
  (* The unconsumed body would otherwise leak memory *)
  let* () = if status <> `OK then Cohttp_lwt.Body.drain_body body else Lwt.return_unit in
  match status with
  | `OK ->
    print_string "OK";
    Lwt.return (response, body)
  | `Permanent_redirect | `Moved_permanently ->
    print_string "Redirect";
    handle_redirect ~enc_path response
  | `Found | `Temporary_redirect ->
    print_string "Temp Redirect\n";
    handle_redirect ~enc_path response
  | `Not_found | `Gone -> Lwt.fail_with "Not found"
  | status ->
    Lwt.fail_with
      (Printf.sprintf "Unhandled status: %s" (Cohttp.Code.string_of_status status))

and handle_redirect ~enc_path response =
  let headers = Cohttp.Response.headers response in
  let location = Cohttp.Header.get headers "location" in
  match location with
  | None -> Lwt.fail_with "Redirection without Location header"
  | Some url ->
    let curl_get = Printf.sprintf "curl \"%s\" > %s \n" url enc_path in
    Printf.printf "Downloading %s\n" enc_path;
    let result = Sys.command curl_get in
    Printf.printf "Result:%d\n" result;
    if result == 0
    then (
      let resp = Cohttp_lwt.Response.make ~status:(Cohttp.Code.status_of_code 200) () in
      let body = Cohttp_lwt.Body.empty in
      Lwt.return (resp, body))
    else
      Lwt.return
        ( Cohttp_lwt.Response.make ~status:(Cohttp.Code.status_of_code 500) ()
        , Cohttp_lwt.Body.empty )
;;
