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

let token_count t = Hashtbl.length t.token_from_subword
let pairs_count t = Hashtbl.length t.rank_from_pair
