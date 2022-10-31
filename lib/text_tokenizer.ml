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

let zip_next subwords =
  let l = Array.length subwords - 1 in
  Array.init (l - 1) (fun i -> subwords.(i), subwords.(i + 1))
;;

let find_min_idx pairs_rank =
  let find_min cidx acc cur =
    let _, elem = acc in
    if cur < elem then cidx, cur else acc
  in
  Base.Array.foldi ~f:find_min ~init:(-1, Base.Int.max_value) pairs_rank
;;

let join_subwords arr idx =
  Base.Array.filter_mapi arr ~f:(fun i el ->
    if i == idx then Some (Base.String.concat [ arr.(i); arr.(i + 1) ]) else Some el)
;;

let get_bpe t _is_verbose word =
  let start = Char.unsafe_chr @@ (Char.code ' ' + 256) in
  let subwords = List.of_seq (String.to_seq word) in
  let subwords = List.cons start subwords in
  let subwords = Array.of_list subwords in
  let subwords = Array.map Base.String.of_char subwords in
  let subwords = ref subwords in
  while Array.length !subwords > 1 do
    let pairs = zip_next !subwords in
    let pairs_rank =
      Array.map
        (fun x ->
          Option.value (Hashtbl.find_opt t.rank_from_pair x) ~default:Base.Int.max_value)
        pairs
    in
    let min_idx, _elem = find_min_idx pairs_rank in
    let pair_to_merge = pairs.(min_idx) in
    let rank_sel = Hashtbl.find_opt t.rank_from_pair pair_to_merge in
    if Option.is_none rank_sel
    then subwords := Array.make 0 String.empty
    else subwords := join_subwords !subwords min_idx
  done;
  Array.to_list !subwords
;;

let tokenize t ~text ~is_verbose =
  let sep_token = Hashtbl.find t.token_from_subword "</s>" in
  let cls_token = Hashtbl.find t.token_from_subword "<s>" in
  let unk_token = Hashtbl.find t.token_from_subword "<unk>" in
  let tokens =
    String.split_on_char ' ' text
    |> List.filter (fun x -> String.length x > 0)
    |> List.map (Base.Fn.compose (get_bpe t is_verbose) String.lowercase_ascii)
    |> List.flatten
    |> List.map (fun x ->
         Option.value (Hashtbl.find_opt t.token_from_subword x) ~default:unk_token)
  in
  List.append (List.cons cls_token tokens) [ sep_token ]
;;
