type t =
  { token_from_subword : (string, int) Hashtbl.t
  ; rank_from_pair : (string * string, int) Hashtbl.t
  }

let make token_from_subword merges =
  let take_two line =
    let parts = String.split_on_char ' ' line in
    List.hd parts, List.hd (List.tl parts)
  in
  let pairs = List.map take_two merges in
  let rank_from_pair = Hashtbl.create (List.length pairs) in
  List.iteri (Base.Fn.flip (Hashtbl.add rank_from_pair)) pairs;
  { token_from_subword; rank_from_pair }
;;

let zip_next subwords =
  let l = Array.length subwords in
  Array.init (l - 1) (fun i -> subwords.(i), subwords.(i + 1))
;;

let find_min_idx pairs_rank =
  let find_min cidx (pidx, elem) cur = if cur < elem then cidx, cur else pidx, elem in
  Base.Array.foldi ~f:find_min ~init:(0, pairs_rank.(0)) pairs_rank
;;

let join_subwords arr idx =
  Base.Array.filter_mapi arr ~f:(fun i el ->
    if i == idx
    then Some (Base.String.concat [ arr.(i); arr.(i + 1) ])
    else if i == idx + 1
    then None
    else Some el)
;;

let get_bpe t is_verbose word =
  let start = Uchar.of_int (Char.code ' ' + 256) in
  let buf = Buffer.create 1 in
  Uutf.Buffer.add_utf_8 buf start;
  let start = Buffer.contents buf in
  let subwords =
    Base.String.to_list word
    |> List.map Base.String.of_char
    |> List.cons start
    |> Array.of_list
    |> ref
  in
  let is_done = ref false in
  while Array.length !subwords > 1 && !is_done == false do
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
    then is_done := true
    else subwords := join_subwords !subwords min_idx
  done;
  if is_verbose
  then Array.iteri (fun i x -> Stdio.printf "subwords:%d|%s\n" i x) !subwords
  else ();
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
  (cls_token :: tokens) @ [ sep_token ]
;;
