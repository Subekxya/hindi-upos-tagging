from pathlib import Path


def read_conllu_sentences(file_path):
    """Read a CoNLL-U file into sentences of word dictionaries"""
    file_path = Path(file_path)
    sentences = []
    current_sentence = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) != 10:
                raise ValueError(f"{file_path}, line {line_number}:\nexpected 10 columns, found {len(columns)}")

            token_id = columns[0]

            #Skip multiword tokens and empty nodes
            if not token_id.isdigit():
                continue

            form = columns[1]
            upos = columns[3]

            if not form or upos == "_":
                raise ValueError(f"{file_path}, line {line_number}:\nmissing FORM or UPOS")

            current_sentence.append({"ID": int(token_id),
                                     "FORM": form,
                                     "UPOS": upos})

        #Include the last sentence if there is no final blank line
        if current_sentence:
                sentences.append(current_sentence)

        return sentences


def count_conllu(file_path):
    """Count sentence and word-token for a CoNLL-U file."""
    sentences = read_conllu_sentences(file_path)
    token_count = sum(len(sentence) for sentence in sentences)

    return {"sentences": len(sentences),
            "word_tokens": token_count}


def tokenize_and_align_labels(sentences, tokenizer, label2id, max_length=128):
    """Tokenize word-level sentences and align UPOS labels to subtokens"""
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sentence in sentences:
        words = [token["FORM"] for token in sentence]
        word_labels = [token["UPOS"] for token in sentence]

        encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)

        words_ids = encoded.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in words_ids:
            if word_id is None or word_id == previous_word_id:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label2id[word_labels[word_id]])

            previous_word_id = word_id

        all_input_ids.append(encoded["input_ids"])
        all_attention_masks.append(encoded["attention_mask"])
        all_labels.append(aligned_labels)

    return {"input_ids": all_input_ids,
             "attention_mask": all_attention_masks,
             "labels": all_labels}