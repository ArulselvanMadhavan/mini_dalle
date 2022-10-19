val http_get_and_follow: Uri.t -> string
  -> (Cohttp.Response.t * Cohttp_lwt.Body.t) Lwt.t
