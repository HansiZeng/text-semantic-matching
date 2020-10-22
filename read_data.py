import json
import gzip
from tqdm import tqdm
import os

import numpy as np
import torch
import torch.nn as nn 

#args.dataset
#args.use_char
#args.use_pos
#args.use_local_feats
#args.use_lower

#args.char_max
#args.qmax = 80
#args.amax = 60

# ============= utils ===============
def dict_from_gzip(path):
    print("Loading {}".format(path))
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())

def dict_to_gzip(dict, path):
    print("Writing to {}".format(path))
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))

class NLIDataset():
    def __init__(self, args):
        self.args = args
        
        self.data_dir = self.args.data_dir
        self.dest_dir = self.args.dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir )

        self.register = {}

        # dictionary
        self.word2idx = {}
        self.char2idx = {}
        self.pos2idx = {}
        self.words = []
        self.chars = []
        self.poss = []

        # dataset 
        self.train_set = None 
        self.valid_set = None 
        self.test_set = None

        self.train_pos = None
        self.valid_pos = None 
        self.test_pos = None 

        self.train_feats = None 
        self.valid_feats = None 
        self.test_feats = None 

        # set up 
        #self.create_examples()

    def load_sets(self):
        """
        - load train, valid, test data
        - load `features` dictionary respectively
        - output information
        """
        envs = dict_from_gzip(self.data_dir + "/" + "env.gz")

        if True:
            self.word2idx = envs["word_index"]
            sorted_word2idx = dict(sorted(self.word2idx.items(), key=lambda item: item[1]))
            self.words = list(sorted_word2idx.keys())
            del sorted_word2idx

            self.train_set = envs["train"]
            self.valid_set = envs["dev"]
            self.test_set = envs["test"]

        if self.args.use_char:
            self.char2idx = envs["char_index"]
            sorted_char2idx = dict(sorted(self.char2idx.items(), key=lambda item: item[1]))
            self.chars = list(sorted_char2idx.keys())
            del sorted_char2idx

        if self.args.use_pos:
            self.pos2idx = envs["pos_index"]
            sorted_pos2idx = dict(sorted(self.pos2idx.items(), key=lambda item: item[1]))
            self.poss = list(sorted_pos2idx.keys())
            del sorted_pos2idx

            pos_env = dict_from_gzip(self.data_dir + "/" + "pos_env.gz")
            self.train_pos = pos_env["train_pos"]
            self.valid_pos = pos_env["dev_pos"]
            self.test_pos = pos_env["test_pos"]

        if self.args.use_local_feats:
            feat_env = dict_from_gzip(self.data_dir + "/" + "feat_env.gz")
            self.train_feats = feat_env["train_feats"]
            self.valid_feats = feat_env["dev_feats"]
            self.test_feats = feat_env["test_feats"]

    def pad_sequence(self, seq, max_len, padding_element=0):
        res_len = max_len - len(seq)
        if res_len >= 0:
            seq = seq + res_len * [padding_element]
        else:
            seq = seq[:max_len]
        return seq

    def register_field(self, id, name):
        self.register[name] = id

    def prepare_dataset(self, data, local_feats=None, pos_feats=None):
        """
        Returns:
            outputs: List of examples, 
                left_words, right_words, left_lengths, right_lengths
                left_chars, right_chars, left_pos, right_pos, left_feats, right_feats, label
        """
        outputs = []

        def word_idxs(words, word_max):
            if self.args.word_lower:
                words = [w.lower() for w in words]
            _ys = [self.word2idx[w] if w in self.word2idx else 1  for w in words]
            _ys = self.pad_sequence(_ys, word_max)
            return _ys

        def char_idxs(words, char_max, word_max):
            _ys = [[self.char2idx[c] for c in word ] for word in words]
            _ys = [self.pad_sequence(_y, char_max, 0) for _y in _ys]
            _ys  = self.pad_sequence(_ys, word_max, [0] * char_max)
            return _ys 

        def pos_idxs(seq, seq_max):
            _ys = [self.pos2idx[x] for x in seq]
            _ys = self.pad_sequence(_ys, seq_max, 0)
            return _ys


        # form words
        left_lengths = [min(len(x[0]), self.args.qmax) for x in tqdm(data)]
        right_lengths = [min(len(x[1]), self.args.amax) for x in tqdm(data)]
        left_words = [ word_idxs(x[0], self.args.qmax) for x in tqdm(data)]
        right_words = [ word_idxs(x[1], self.args.amax) for x in tqdm(data)]
        left_lengths = np.array(left_lengths)
        right_lengths = np.array(right_lengths)
        left_words = np.array(left_words)
        right_words = np.array(right_words)
        print("shape of left words: {}, right words: {}, left lengths: {}, right lengths: {}".format(
            left_words.shape, right_words.shape, left_lengths.shape, right_lengths.shape
        ))
        outputs = [left_words, right_words, left_lengths, right_lengths]
        self.register_field(0, "left_words")
        self.register_field(1, "right_words")
        self.register_field(2, "left_lengths")
        self.register_field(3, "right_lengths")

        # form chars 
        if self.args.use_char:
            left_chars = [ char_idxs(x[0], self.args.char_max, self.args.qmax) for x in tqdm(data)]
            right_chars = [ char_idxs(x[1], self.args.char_max, self.args.amax) for x in tqdm(data)]
            left_chars = np.array(left_chars)
            right_chars = np.array(right_chars)
            print("left chars shape: {}, right chars shape: {}".format(left_chars.shape, right_chars.shape))
            outputs.append(left_chars)
            self.register_field(len(outputs), "left_chars")
            outputs.append(right_chars)
            self.register_field(len(outputs), "right_chars")
        
        # form pos 
        if self.args.use_pos and pos_feats is not None:
            left_pos = [pos_idxs(example[0], self.args.qmax) for example in tqdm(pos_feats)]
            right_pos = [pos_idxs(example[1], self.args.amax) for example in tqdm(pos_feats)]
            left_pos = np.array(left_pos)
            right_pos = np.array(right_pos)
            print("shape of left pos: {}, right pos: {}".format(left_pos.shape, right_pos.shape))
            outputs.append(left_pos)
            self.register_field(len(outputs), "left_pos")
            outputs.append(right_pos)
            self.register_field(len(outputs), "right_pos")

        # form local feats
        if self.args.use_local_feats and local_feats is not None:
            left_feats = [self.pad_sequence(example[0], self.args.qmax, 0) for example in tqdm(local_feats)]
            right_feats = [self.pad_sequence(example[1], self.args.amax, 0) for example in tqdm(local_feats)]
            left_feats = np.array(left_feats).reshape(-1, self.args.qmax, 1)
            right_feats = np.array(right_feats).reshape(-1, self.args.amax, 1)
            print("shape of left feats: {}, right feats: {}".format(left_feats.shape, right_feats.shape))
            outputs.append(left_feats)
            self.register_field(len(outputs), "left_feats")
            outputs.append(right_feats)
            self.register_field(len(outputs), "right_feats")

        # last to add label
        labels = [x[2] for x in tqdm(data)]
        labels = np.array(labels)
        print("shape of labels: {}".format(labels.shape))
        outputs.append(labels)
        self.register_field(len(outputs), "labels")

        outputs = zip(*outputs)
        outputs = list(outputs)
        original_len = len(outputs)
        outputs = [example for example in outputs if example[2] > 0 and example[3] > 0]
        print("filterd={}".format(original_len - len(outputs)))

        return outputs

    def create_examples(self):
        self.load_sets()
        self._train_examples = self.prepare_dataset(self.train_set, local_feats=self.train_feats, pos_feats=self.train_pos)
        self._valid_examples = self.prepare_dataset(self.valid_set, local_feats=self.valid_feats, pos_feats=self.valid_pos)
        self._test_examples = self.prepare_dataset(self.test_set, local_feats=self.test_feats, pos_feats=self.test_pos)

        # save examples
        #self.save_examples("train")
        #self.save_examples("valid")
        #self.save_examples("test")

    @property
    def train_examples(self):
        return self._train_examples 

    @property
    def valid_examples(self):
        return self._valid_examples

    @property
    def test_examples(self):
        return self._test_examples

    def save_examples(self, set_name, compress="pickle"):
        if set_name == "train":
            _examples = self._train_examples
        elif set_name == "valid":
            _examples = self._valid_examples
        elif set_name == "test":
            _examples = self._test_examples
        else:
            raise ValueError("set name {} is not defined".format(set_name))

        if compress == "gzip":
            import pandas as pd 
            _examples_str = pd.Series(_examples).to_json(orient="values")
            path = self.dest_dir + "/" + f"{set_name}_example.gz"
            with gzip.open(path, "wt") as f:
                f.write(_examples_str)
            del _examples_str
        elif compress == "pickle":
            import pickle
            path = self.dest_dir + "/" + f"{set_name}_example.pkl"
            with open(path, "wb") as f:
                pickle.dump(_examples, f)
        else:
            raise ValueError("compress type {} is not predefined".format(compress))

    def save_all_examples(self):
        self.save_examples("train")
        self.save_examples("valid")
        self.save_examples("test")
    
    def load_examples(self, set_name, compress="pickle"):
        if compress == "pickle":
            suffix = ".pkl" 
        elif compress == "gizp":
            suffix = ".gz"
        else:
            raise ValueError("suffix name {} is not defined".format(compress))

        if set_name == "train":
            path = self.dest_dir + "/" + "train_example" + suffix
        elif set_name == "valid":
            path = self.dest_dir + "/" + "valid_example" + suffix
        elif set_name == "test":
            path = self.dest_dir + "/" + "test_example" + suffix
        else:
            raise ValueError("set name {} is not defined".format(set_name))
        
        if compress == "gzip":
            with gzip.open(path, "rt") as f:
                _examples = json.loads(f.read())
            for i, exp in enumerate(_examples):
                for j, e in enumerate(exp):
                    e = np.array(e) if type(e) is list else e
                    _examples[i][j] = e 
        elif compress == "pickle":
            import pickle
            with open(path, "rb") as f:
                _examples = pickle.load(f)
        else:
            raise ValueError("suffix name {} is not defined".format(compress))

        
        if set_name == "train":
            self._train_examples = _examples
            return self.train_examples
        elif set_name == "valid":
            self._valid_examples = _examples
            return self.valid_examples
        elif set_name == "test":
            self._test_examples = _examples
            return self.test_examples
        else:
            raise ValueError("set name {} is not defined".format(set_name))
            
if __name__ == "__main__":
    config_file = "data_default.json"
    from utils import parse_args
    args = parse_args(config_file)

    nli_dataset = NLIDataset(args)

    # part for writing and saving exmamples
    nli_dataset.create_examples()
    nli_dataset.save_all_examples()

    # part for reading examples
    print("start loading examples ...")

    train_examples = nli_dataset.load_examples("train")
    #valid_examples = nli_dataset.load_examples("valid")
    print("shape of valid examples {}".format(len(train_examples)))
    for i, exp in enumerate(train_examples):
        for e in exp:
            if type(e) is np.ndarray:
                print(e.shape, e.dtype)
            else:
                print(e)

        if i == 1:
            break
    