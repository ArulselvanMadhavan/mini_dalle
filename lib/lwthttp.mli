val http_get_and_follow
  :  max_redirects:int
  -> Uri.t
  -> (Cohttp.Response.t * Cohttp_lwt.Body.t) Lwt.t
