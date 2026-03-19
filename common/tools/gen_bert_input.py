"""
Copyright 2024 The HAKES Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# %%
import numpy as np
import gluonnlp
import logging
import argparse

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
import struct


# %%
class BertInputTransformer:
    def __init__(self, vocab_file, max_seq_length):
        with open(vocab_file, "r") as f:
            vocabulary = gluonnlp.vocab.BERTVocab.from_json(f.read())
        tokenizer = gluonnlp.data.BERTTokenizer(vocabulary, lower=True)
        self.transform = gluonnlp.data.BERTSentenceTransform(
            tokenizer, max_seq_length=max_seq_length, pair=False
        )

    def apply(self, text):
        sample = self.transform([text])
        words = sample[0]
        valid_len = sample[1]
        segments = sample[2]
        return valid_len, words, segments


def write_bert_input(max_seq_length, valid_len, words, segments, output_name):
    with open(output_name, "wb") as f:
        f.write(struct.pack("<i", max_seq_length))
        f.write(struct.pack("<f", valid_len))
        f.write(np.ascontiguousarray(words, dtype="<f").tobytes())
        f.write(np.ascontiguousarray(segments, dtype="<f").tobytes())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BERT input")
    parser.add_argument("--vocab_file", type=str, required=True, help="Vocabulary file")
    parser.add_argument("--input_text", type=str, required=True, help="Input text")
    parser.add_argument(
        "--max_seq_length", type=int, required=True, help="Sequence length"
    )
    parser.add_argument(
        "--output_name", type=str, required=True, help="Output file name"
    )
    return parser.parse_args()


def run(vocab_file, input_text, max_seq_length, output_name):
    transformer = BertInputTransformer(vocab_file, max_seq_length)
    valid_len, words, segments = transformer.apply(input_text)
    write_bert_input(max_seq_length, valid_len, words, segments, output_name)


if __name__ == "__main__":
    args = parse_args()
    run(args.vocab_file, args.input_text, args.max_seq_length, args.output_name)

# Usage
# python gen_bert_input.py --vocab_file  ../data/tvm-embed-bert/bert/book_corpus_wiki_en_uncased-a6607397.vocab --input_text "hello world!" --max_seq_length 128 --output_name bert_input.bin

# expected behavior: hello world! -> valid_length = 5, words = [2, 7592,  2088, 999, 3] the reset 1, segments = [0 0 0 0 0] the rest 0. After embedding: [-0.897565, -0.330401, -0.769419, ... ,-0.655948, -0.619980, 0.909517]
