from datasets import load_dataset, Dataset


def main():
    raw_dataset = load_dataset()
    tq = raw_dataset.map(lambda e: tokenizer(e, truncation=True, padding="max_length"), input_columns=["query"])
    tpd = raw_dataset.map(lambda e: tokenizer(e, truncation=True, padding="max_length"), input_columns=["positive_doc"])
    tnd = raw_dataset.map(lambda e: tokenizer(e, truncation=True, padding="max_length"), input_columns=["negative_doc"])
    train_dataset = Dataset()
    train_dataset.add_column("t_query", tq)
    train_dataset.add_column("t_pos_doc", tpd)
    train_dataset.add_column("t_neg_doc", tnd)
