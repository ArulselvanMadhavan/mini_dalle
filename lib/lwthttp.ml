let sample = "https://cdn-lfs.huggingface.co/repos/4c/18/4c187610eb96c173cde1753e1a5df664940a06e49946e34b009b0e59fce89d0c/76e7797b19625122e21138556e06b78cc84686ed5f5e5c42aa53c81ffb2f4bb8?response-content-disposition=attachment%3B%20filename%3D%22encoder.pt%22&Expires=1666268611&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzRjLzE4LzRjMTg3NjEwZWI5NmMxNzNjZGUxNzUzZTFhNWRmNjY0OTQwYTA2ZTQ5OTQ2ZTM0YjAwOWIwZTU5ZmNlODlkMGMvNzZlNzc5N2IxOTYyNTEyMmUyMTEzODU1NmUwNmI3OGNjODQ2ODZlZDVmNWU1YzQyYWE1M2M4MWZmYjJmNGJiOD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPWF0dGFjaG1lbnQlM0IlMjBmaWxlbmFtZSUzRCUyMmVuY29kZXIucHQlMjIiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NjYyNjg2MTF9fX1dfQ__&Signature=FxXR0ZuQugZz42JPFn-Ir2A3IJg4sbu9~dP8vf5O-yp5LXG9nNmo4~8TOoa-Zf8bdn3N-0cqQb8xN7pJGL58Oi3D8EvWgMsTY5-fug-5LCjaUWKjg2aYDAvg5cK8Q72MqIp1jkKPsje6fQSslM7ZugA1FyTv00POKPQHmfmSqM-NUr~PQ1wPnIWR2I5rZWq276ZYty2naLNJDAJh1wbbf6SeADjV2W~cSBtEkgShgXc7aJ2oIaUW3P4KY18Raea-rQgw62yO6cVawO4PTfUw6PXig7ccGNWfTbFFhizV0jVsc8D18CoKBOgtFzyr5j7U-idw21qVpclvZDwS-RMYVw__&Key-Pair-Id=KVTP0A1DKRTAX"
  
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
    print_string "Temp Redirect\n"; handle_redirect ~permanent:false ~max_redirects request_uri response
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
      (* let sample = Uri.pct_decode sample in
       * print_string sample; *)
      let uri = Uri.of_string sample in
      let uri = Uri.canonicalize uri in
      let* () =
        if permanent
        then
          Logs_lwt.warn (fun m ->
            m "Permanent redirection from %s to %s" (Uri.to_string request_uri) url)
        else Lwt.return_unit
      in
      Printf.printf "\n%s\n" @@ Uri.to_string uri;
      http_get_and_follow uri ~max_redirects:(max_redirects - 1))
;;
