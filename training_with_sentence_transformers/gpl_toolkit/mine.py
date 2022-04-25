import json
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import torch
from easy_elasticsearch import ElasticSearchBM25
import tqdm
import numpy as np
import os
import logging
import argparse
import time

from splade_vocab.models import Splade

logger = logging.getLogger(__name__)


class NegativeMiner(object):
    def __init__(
        self,
        generated_path,
        prefix,
        retrievers={
            "bm25": "bm25",
            "sbert": "msmarco-distilbert-base-v3",
            "ce": "msmarco-MiniLM-L-6-v3",
            "splade": "path",
        },
        nneg=50,
        sep=" ",
    ):
        self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(generated_path, prefix=prefix).load(
            split="train"
        )
        self.output_path = os.path.join(generated_path, "hard-negatives.jsonl")
        self.sep = sep
        self.retrievers = retrievers
        if "bm25" in retrievers:
            assert nneg <= 10000, "Only `negatives_per_query` <= 10000 is acceptable by Elasticsearch-BM25"

        self.nneg = nneg
        if nneg > len(self.corpus):
            logger.warning("`negatives_per_query` > corpus size. Please use a smaller `negatives_per_query`")
            self.nneg = len(self.corpus)

    def _get_doc(self, did):
        return self.sep.join([self.corpus[did]["title"], self.corpus[did]["text"]])

    def _mine_sbert(self, model_name):
        logger.info(f"Mining with {model_name}")
        result = {}
        sbert = SentenceTransformer(model_name)
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        batch_size = 128
        doc_embs = sbert.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), batch_size):
            qid_batch = qids[start : start + batch_size]
            qemb_batch = sbert.encode(
                queries[start : start + batch_size],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
            _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result

    def _mine_splade(self, model_name):
        logger.info(f"Mining with {model_name}")
        result = {}
        splade = Splade(model_name)
        splade.eval()
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        d_batch_size = 64
        q_batch_size = 1024
        doc_embs = splade.encode_sentence_bert(
            docs,
            batch_size=d_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            convert_to_spsparse=True,
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), q_batch_size):
            qid_batch = qids[start : start + q_batch_size]
            qemb_batch = splade.encode_sentence_bert(
                queries[start : start + q_batch_size],
                batch_size=q_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                convert_to_spsparse=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
                maxlen=64,
            )
            score_mtrx = qemb_batch @ doc_embs.T  # (qsize, dsize)
            score_mtrx = torch.from_numpy(score_mtrx.toarray())
            _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result

    def _mine_splade_dense(self, model_name):
        logger.info(f"Mining with {model_name}")
        result = {}
        splade = Splade(model_name)
        splade.eval()
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        q_batch_size = 1536
        d_batch_size = 64
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), q_batch_size):
            qid_batch = qids[start : start + q_batch_size]
            qemb_batch = splade.encode_sentence_bert(
                queries[start : start + q_batch_size],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=False,
                maxlen=64,
            )
            score_mtrxs = []
            for d_start in tqdm.trange(0, len(docs), d_batch_size):
                doc_batch = docs[d_start : d_start + d_batch_size]
                doc_embs = splade.encode_sentence_bert(
                    doc_batch,
                    batch_size=d_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                )
                score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
                score_mtrxs.append(score_mtrx)
            score_mtrx = torch.cat(score_mtrxs, dim=1).cpu()
            _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result

    def _mine_bm25(self):
        logger.info(f"Mining with bm25")
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=self.nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())
            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)
            result[qid] = neg_dids
        return result

    def run(self):
        hard_negatives = {}
        for retriever_name, retriever_path in self.retrievers.items():
            if retriever_name == "bm25":
                hard_negatives[retriever_name] = self._mine_bm25()
            elif retriever_name == "splade":
                hard_negatives[retriever_name] = self._mine_splade(retriever_path)
            else:
                hard_negatives[retriever_name] = self._mine_sbert(retriever_path)

        logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {"qid": qid, "pos": list(pos_dids.keys()), "neg": {k: v[qid] for k, v in hard_negatives.items()}}
            result_jsonl.append(line)

        logger.info(f"Saving data to {self.output_path}")
        with open(self.output_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_path")
    args = parser.parse_args()

    miner = NegativeMiner(args.generated_path, "qgen")
    miner.run()
