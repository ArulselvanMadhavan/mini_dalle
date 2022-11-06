(lang dune 3.4)

(name mini_dalle)

(generate_opam_files true)

(source
 (github ArulselvanMadhavan/mini_dalle))

(authors "Arulselvan Madhavan")

(maintainers "Arulselvan Madhavan")

(license Apache License 2.0)

(documentation https://github.com/ArulselvanMadhavan/mini_dalle/README.org)

(package
 (name mini_dalle)
 (synopsis "Mini Dalle in OCaml")
 (description "Mini Dalle in OCaml")
 (depends ocaml dune)
 (tags
  (topics "deep learning"))
 (depends
  (ocaml (>= 4.14.0))
  (ocamlfind :build)
  (torch)
  (npy)
  (base)
  (cohttp-lwt-unix)
  (yojson)
  (uutf)
  (ppx_yojson_conv))
 (depopts utop))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
