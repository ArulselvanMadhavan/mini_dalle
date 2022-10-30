type t =
  { token_from_subword : (string, int) Hashtbl.t
  ; rank_from_pair : (string * string, int) Hashtbl.t
  }

let make token_from_subword merges =
  let pairs =
    List.map
      (fun pair ->
        let pairs = String.split_on_char ' ' pair in
        List.hd pairs, List.hd pairs)
      merges
  in
  let rank_from_pair = Hashtbl.create (List.length pairs) in
  List.iteri (fun idx elem -> Hashtbl.add rank_from_pair elem idx) pairs;
  { token_from_subword; rank_from_pair }
;;

let get_bpe _is_verbose word = [ word ]

let tokenize t ~text ~is_verbose =
  let sep_token = Hashtbl.find t.token_from_subword "</s>" in
  let cls_token = Hashtbl.find t.token_from_subword "<s>" in
  let unk_token = Hashtbl.find t.token_from_subword "<unk>" in
  
  let tokens = String.split_on_char ' ' text
  |> List.filter (fun x -> String.length x > 0)
  |> List.map (Base.Fn.compose (get_bpe is_verbose) String.lowercase_ascii)
  |> List.flatten
  |> List.map (fun x -> Option.value (Hashtbl.find_opt t.token_from_subword x) ~default:unk_token) in
  List.append (List.cons cls_token tokens) [sep_token]
;;
