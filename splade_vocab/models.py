# FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
# Original License APACHE2
# And From SPLADE(https://github.com/naver/splade)
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

import os
import json
import logging
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import numpy as np
import scipy as sp
import torch
from numpy import ndarray
from scipy.sparse import csr_matrix
from torch import Tensor, nn
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers.file_utils import ModelOutput

try:
    import sentence_transformers
    from sentence_transformers.util import batch_to_device
    from sentence_transformers.models import WordWeights
except ImportError:
    print("Import Error: could not load sentence_transformers... proceeding")
logger = logging.getLogger(__name__)


IDF_FILE_NAME = "weights.json"


class BEIRSbert:
    def __init__(self, model, tokenizer, max_length=256, sep=" "):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.q_model = model
        self.doc_model = model

        self.sep = sep

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [
            (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus
        ]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)


class BEIRSpladeModelIDF:
    def __init__(self, model, tokenizer, idf, max_length=256, sqrt=True):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        self.idf = self._init_idf(idf, sqrt)
        print(self.idf.shape)

    def _init_idf(self, idf, sqrt):
        idf_vec = np.ones(len(self.tokenizer.vocab))
        for k, v in idf.items():
            if v == 0.0:
                continue
            if sqrt:
                idf_vec[k] = np.sqrt(v)
            else:
                idf_vec[k] = v
        return idf_vec

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        X = self.model.encode_sentence_bert(sentences, maxlen=self.max_length)
        X *= self.idf
        return X


class BEIRSpladeModelBM25:
    def __init__(self, model, tokenizer, idf, doc_len_ave, max_length=256, bm25_k1=0.9, bm25_b=0.4):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        self.idf = self._init_idf(idf)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.doc_len_ave = doc_len_ave

    def _init_idf(self, idf):
        idf_vec = np.ones(len(self.tokenizer.vocab))
        for k, v in idf.items():
            if v == 0.0:
                continue
            idf_vec[k] = v
        return idf_vec

    def bm25_tf(self, tf, doc_lens):
        nume = tf * (1 + self.bm25_k1)
        denom = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_lens / self.doc_len_ave)
        return nume / denom

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        X = self.model.encode_sentence_bert(sentences, maxlen=self.max_length)
        input_tfs = np.ones(X.shape)
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)
        doc_lens = []
        for i, (input_tokens, att_mask) in enumerate(zip(i_sentences["input_ids"], i_sentences["attention_mask"])):
            tf = Counter(input_tokens)
            doc_lens.append(np.sum(att_mask))
            for k, v in tf.items():
                input_tfs[i, k] *= v
        doc_lens = np.ravel(doc_lens)[:, np.newaxis]
        tf_weight = self.bm25_tf(input_tfs, doc_lens)
        X *= tf_weight * self.idf
        return X


class BEIRSpladeModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        return self.model.encode_sentence_bert(sentences, maxlen=self.max_length)


class BEIRSpladeDocModel(BEIRSpladeModel):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            X[i, i_query] += 1
        return X


class BEIRSpladeDocModelIDF(BEIRSpladeModelIDF):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            X[i, i_query] += 1
        return X


class BEIRSpladeDocModelBM25(BEIRSpladeModelBM25):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            X[i, i_query] += 1
        return X


class BEIRSpladeQueryModel(BEIRSpladeModel):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_sentences):
            X[i, i_query] += 1
        return X


class BEIRSpladeQueryModelIDF(BEIRSpladeModelIDF):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_sentence in enumerate(i_sentences):
            X[i, i_sentence] += self.idf[i_sentence]
        return X


class BEIRSpladeQueryModelBM25(BEIRSpladeModelBM25):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        input_tfs = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)
        doc_lens = []
        for i, (input_tokens, att_mask) in enumerate(zip(i_sentences["input_ids"], i_sentences["attention_mask"])):
            tf = Counter(input_tokens)
            doc_lens.append(np.sum(att_mask))
            for k, v in tf.items():
                input_tfs[i, k] *= v
        doc_lens = np.ravel(doc_lens)[:, np.newaxis]
        tf_weight = self.bm25_tf(input_tfs, doc_lens)
        tf_weight *= self.idf
        return tf_weight


class BEIRSpladeEMModel(BEIRSpladeModel):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        mask = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            mask[i, i_query] += 1
        X *= mask
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = self.model.encode_sentence_bert(sentences, maxlen=self.max_length)
        mask = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_sentence in enumerate(i_sentences):
            i_sentence = list(set(i_sentence))
            mask[i, i_sentence] += 1
        X *= mask
        return X


class BEIRSpladeEMModelIDF(BEIRSpladeModelIDF):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        mask = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            mask[i, i_query] += 1
        X *= mask
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = self.model.encode_sentence_bert(sentences, maxlen=self.max_length)
        mask = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_sentence in enumerate(i_sentences):
            i_sentence = list(set(i_sentence))
            mask[i, i_sentence] += self.idf[i_sentence]
        X *= mask
        return X


class BEIRSpladeEMModelBM25(BEIRSpladeModelBM25):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(queries, maxlen=self.max_length)
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        mask = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            mask[i, i_query] += 1
        X *= mask
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        X = self.model.encode_sentence_bert(sentences, maxlen=self.max_length)
        input_tfs = np.zeros(X.shape)
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)
        doc_lens = []
        for i, (input_tokens, att_mask) in enumerate(zip(i_sentences["input_ids"], i_sentences["attention_mask"])):
            tf = Counter(input_tokens)
            doc_lens.append(np.sum(att_mask))
            for k, v in tf.items():
                input_tfs[i, k] += v

        doc_lens = np.ravel(doc_lens)[:, np.newaxis]
        tf_weight = self.bm25_tf(input_tfs, doc_lens)
        X *= tf_weight * self.idf
        return X


class BEIRSpladeEMDocModel(BEIRSpladeEMModel):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            X[i, i_query] += 1
        return X


class BEIRSpladeEMDocModelIDF(BEIRSpladeEMModelIDF):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            X[i, i_query] += 1
        return X


class BEIRSpladeEMDocModelBM25(BEIRSpladeEMModelBM25):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        i_queries = self.tokenizer(queries, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(queries), len(self.tokenizer.get_vocab())))
        for i, i_query in enumerate(i_queries):
            i_query = list(set(i_query))
            X[i, i_query] += 1
        return X


class BEIRSpladeEMQueryModel(BEIRSpladeEMModel):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_sentence in enumerate(i_sentences):
            i_sentence = list(set(i_sentence))
            X[i, i_sentence] += 1
        return X


class BEIRSpladeEMQueryModelIDF(BEIRSpladeEMModelIDF):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        X = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        for i, i_sentence in enumerate(i_sentences):
            i_sentence = list(set(i_sentence))
            X[i, i_sentence] += self.idf[i_sentence]
        return X


class BEIRSpladeEMQueryModelBM25(BEIRSpladeEMModelBM25):
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        input_tfs = np.zeros((len(sentences), len(self.tokenizer.get_vocab())))
        i_sentences = self.tokenizer(sentences, add_special_tokens=False)
        doc_lens = []
        for i, (input_tokens, att_mask) in enumerate(zip(i_sentences["input_ids"], i_sentences["attention_mask"])):
            tf = Counter(input_tokens)
            doc_lens.append(np.sum(att_mask))
            for k, v in tf.items():
                input_tfs[i, k] += v
        doc_lens = np.ravel(doc_lens)[:, np.newaxis]
        tf_weight = self.bm25_tf(input_tfs, doc_lens)
        tf_weight *= self.idf
        return tf_weight


class Splade(nn.Module):
    def __init__(self, model_type_or_dir, lambda_d=0.0008, lambda_q=0.0006, **kwargs):
        super().__init__()
        if os.path.exists(os.path.join(model_type_or_dir, "0_MLMTransformer")):
            print("path", model_type_or_dir)
            model_type_or_dir = os.path.join(model_type_or_dir, "0_MLMTransformer")

        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)

        self.loss_func = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(
        self,
        inputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vecs = [self.encode(**input_) for input_ in inputs]
        q_vecs = vecs[0]
        d_vecs = torch.cat(vecs[1:])
        scores = torch.mm(q_vecs, d_vecs.T)
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        loss = self.loss_func(scores, labels)
        loss += self.lambda_d * self.FLOPS(d_vecs)
        loss += self.lambda_q * self.FLOPS(q_vecs)
        return RankTrainOutput(loss=loss, scores=scores)

    def encode(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        vec = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1).values

        return vec

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode_sentence_bert(
        self,
        sentences: Union[str, List[str], List[int]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_spsparse: bool = False,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        maxlen: int = 512,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_numpy: If true, the output is a list of sparse vectors. It is necessary convert_to_numpy is True.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == "token_embeddings":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            # features = tokenizer(sentences_batch)
            # print(sentences_batch)
            features = self.tokenizer(
                sentences_batch,
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="only_first",  # truncates to self.max_length
                max_length=maxlen,
                return_attention_mask=True,
                return_tensors="pt",
            )
            # print(features)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.encode(**features)
                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0 : last_mask_id + 1])
                else:  # Sentence embeddings
                    # embeddings = out_features[output_value]
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu().numpy()
                        if convert_to_spsparse:
                            embeddings = csr_matrix(embeddings)
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            if convert_to_spsparse:
                all_embeddings = sp.sparse.vstack(all_embeddings)
            else:
                all_embeddings = np.vstack(all_embeddings)
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings


@dataclass
class RankTrainOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: torch.FloatTensor = None


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class BM25Weight(WordWeights):
    def __init__(
        self,
        vocab: List[str],
        word_weights: Dict[str, float],
        doc_len_ave: float,
        bm25_k1: float = 0.9,
        bm25_b: float = 0.4,
        unknown_word_weight: float = 1,
    ):
        super().__init__(vocab, word_weights, unknown_word_weight)
        self.doc_len_ave = doc_len_ave
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def bm25_tf(self, tf, doc_lens):
        nume = tf * (1 + self.bm25_k1)
        denom = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_lens / self.doc_len_ave)
        return nume / denom

    def forward(self, features: Dict[str, Tensor]):
        attention_mask = features["attention_mask"]
        token_embeddings = features["token_embeddings"]

        input_tfs = []
        for input_tokens, att_mask in zip(features["input_ids"], features["attention_mask"]):
            tf = Counter(input_tokens.tolist())
            input_tfs.append(torch.tensor([tf[t.item()] for t in input_tokens]).unsqueeze(0))
        input_tfs = torch.cat(input_tfs).to(att_mask.device)
        input_tfs *= features["attention_mask"]
        doc_lens = torch.sum(features["attention_mask"], dim=1).unsqueeze(-1)
        tf_weight = self.bm25_tf(input_tfs, doc_lens)

        # Compute a weight value for each token
        token_weights_raw = self.emb_layer(features["input_ids"]).squeeze(-1)
        token_weights = tf_weight * token_weights_raw * attention_mask.float()
        token_weights_sum = torch.sum(token_weights, 1)

        # Multiply embedding by token weight value
        token_weights_expanded = token_weights.unsqueeze(-1).expand(token_embeddings.size())
        token_embeddings = token_embeddings * token_weights_expanded

        features.update({"token_embeddings": token_embeddings, "token_weights_sum": token_weights_sum})
        return features


class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.config_keys = ["word_embedding_dimension"]

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        sentence_embedding = torch.max(
            torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1), dim=1
        ).values
        features.update({"sentence_embedding": sentence_embedding})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)


class MLMTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Dict = {},
        cache_dir: Optional[str] = None,
        tokenizer_args: Dict = {},
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        weights: Dict[int, float] = None,
    ):
        super(MLMTransformer, self).__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = torch.nn.DataParallel(
            AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args
        )
        self.pooling = torch.nn.DataParallel(Splade_Pooling(self.get_word_embedding_dimension()))

        if weights is not None:
            vocab_weight = torch.ones(config.vocab_size)
            for i, w in weights.items():
                vocab_weight[int(i)] = w

            vocab_weight = torch.sqrt(vocab_weight)

            self.vocab_weight = nn.Parameter(vocab_weight)
        else:
            self.vocab_weight = None

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "MLMTransformer({}) with Transformer model: {} ".format(
            self.get_config_dict(), self.auto_model.__class__.__name__
        )

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if self.auto_model.module.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        features = self.pooling(features)

        if self.vocab_weight is not None:
            features["sentence_embedding"] *= self.vocab_weight.unsqueeze(0)

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.module.config.vocab_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length
            )
        )
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        if self.vocab_weight is not None:
            with open(os.path.join(output_path, IDF_FILE_NAME), "w") as fOut:
                weight = {}
                for i, w in enumerate(self.vocab_weight):
                    weight[i] = w.item()
                json.dump(weight, fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)

        weight_path = os.path.join(input_path, IDF_FILE_NAME)
        if os.path.exists(weight_path):
            with open(weight_path) as f:
                weight = json.load(f)
        else:
            weight = None

        return MLMTransformer(model_name_or_path=input_path, weight=weight, **config)
