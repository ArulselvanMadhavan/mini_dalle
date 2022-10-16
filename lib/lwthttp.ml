let rec http_get_and_follow ~max_redirects uri =
  let open Lwt.Syntax in
  let* ans = Cohttp_lwt_unix.Client.get uri in
  follow_redirect ~max_redirects uri ans

and follow_redirect ~max_redirects request_uri (response, body) =
  let open Lwt.Syntax in
  let status = Cohttp.Response.status response in
  (* The unconsumed body would otherwise leak memory *)
  let* () = if status <> `OK then Cohttp_lwt.Body.drain_body body else Lwt.return_unit in
  match status with
  | `OK -> print_string "OK";Lwt.return (response, body)
  | `Permanent_redirect | `Moved_permanently ->
    print_string "Redirect"; handle_redirect ~permanent:true ~max_redirects request_uri response
  | `Found | `Temporary_redirect ->
    print_string "Temp Redirect"; handle_redirect ~permanent:false ~max_redirects request_uri response
  | `Not_found | `Gone -> Lwt.fail_with "Not found"
  | status ->
    Lwt.fail_with
      (Printf.sprintf "Unhandled status: %s" (Cohttp.Code.string_of_status status))

and handle_redirect ~permanent ~max_redirects request_uri response =
  if max_redirects <= 0
  then Lwt.fail_with "Too many redirects"
  else (
    let headers = Cohttp.Response.headers response in
    let location = Cohttp.Header.get headers "location" in
    match location with
    | None -> Lwt.fail_with "Redirection without Location header"
    | Some url ->
      let open Lwt.Syntax in
      let uri = Uri.of_string url in
      let* () =
        if permanent
        then
          Logs_lwt.warn (fun m ->
            m "Permanent redirection from %s to %s" (Uri.to_string request_uri) url)
        else Lwt.return_unit
      in
      Printf.printf "%s\n" @@ Uri.to_string uri;
      http_get_and_follow uri ~max_redirects:(max_redirects - 1))
;;
