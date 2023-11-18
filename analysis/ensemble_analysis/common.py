import json
from collections import defaultdict
from pathlib import Path

MIN_DISCOUNT = 1e-3

def get_result_colbert(trec_path: Path, dataset_dir: Path) -> dict:
    id2qid_path = dataset_dir / "idx2qid.json"
    id2did_path = dataset_dir / "idx2did.json"
    with id2qid_path.open(mode="r") as f:
        id2qid = json.load(f)
    
    with id2did_path.open(mode="r") as f:
        id2did = json.load(f)
                          
    result = defaultdict(dict)
    with trec_path.open(mode="r") as f:
        for line in f:
            i_qid, i_did, rank, score = line.strip().split("\t")
            qid = id2qid[i_qid]
            did = id2did[i_did]
            result[qid][did] = float(score)
    return result

def get_result(path: Path) -> dict:
    with path.open(mode="r") as f:
        result = json.load(f)
    return list(result.values())[0]

def get_result_bm25(path: Path) -> dict:
    with path.open(mode="r") as f:
        result = json.load(f)
    return result

def add_result_org(result1: dict, result2: dict, top_k: int = 100) -> dict:
    new_result = {}
    min_score = {}
    for qid, d2score in result1.items():
        s_d2score = sorted(d2score.items(), key=lambda x: -x[1])[:top_k]
        new_result[qid] = dict()
        min_score[qid] = s_d2score[-1][1]
        for did, score in s_d2score:
            new_result[qid][did] = score
            
    for qid, d2score in result2.items():
        s_d2score = sorted(d2score.items(), key=lambda x: -x[1])[:top_k]
        if qid not in new_result:
            new_result[qid] = dict(s_d2score)
            continue
        for did, score in s_d2score:
            if did not in new_result[qid]:
                new_result[qid][did] = min_score[qid]
            new_result[qid][did] += score 
    return new_result

def add_result(bm25_result, dense_result, top_k=100):
    result = {}
    all_qids = set(list(bm25_result.keys()))
    all_qids |= set(list(dense_result.keys()))
    for qid in all_qids:
        d_result1 = bm25_result.get(qid, None)
        d_result2 = dense_result.get(qid, None)
        if not d_result1 and not d_result2:
            continue
        elif not d_result2:
            d_result1 = sorted(d_result1.items(), key=lambda x: -x[1])[:top_k]
            result[qid] = {k: v for k, v in d_result1}
            continue
        elif not d_result1:
            d_result2 = sorted(d_result2.items(), key=lambda x: -x[1])[:top_k]
            result[qid] = {k: v for k, v in d_result2}
            continue
        d_result1 = {k: v for k, v in sorted(d_result1.items(), key=lambda x: -x[1])[:top_k]}
        d_result2 = {k: v for k, v in sorted(d_result2.items(), key=lambda x: -x[1])[:top_k]}
        # all_dids = set([did for did, _ in d_result1]) | set([did for did, _ in d_result2])
        all_dids = set(list(d_result1.keys())) | set(list(d_result2.keys()))
        result[qid] = {}
        try:
            min_score1 = sorted(d_result1.values())[0] - MIN_DISCOUNT
            # min_score1 = d_result1[-1][1] - MIN_DISCOUNT
        except:
            print(qid, d_result1)
            raise ValueError()
        min_score2 = sorted(d_result2.values())[0] - MIN_DISCOUNT
        # min_score2 = d_result2[-1][1] - MIN_DISCOUNT
        for did in all_dids:
            d_score1 = d_result1.get(did, min_score1)
            d_score2 = d_result2.get(did, min_score2)
            result[qid][did] = d_score1 + d_score2
    return result
