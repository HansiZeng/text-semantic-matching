import pickle 
import argparse
import json
import os
import time
import re
from collections import defaultdict
import csv
import gzip

import torch 
import torch.nn as nn
from torch import LongTensor
import numpy as np

from utils import correct_instance_count, parse_args
from experiment import Experiment, parse_args

"""
Read data from data/processed/
each example has 11 fields:
p, q, p_len, q_len, 
q_char, q_char,
p_pos, q_pos
p_feat, q_feat
label
"""

class AvgMeters(object):
    def __init__(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.
    
    def update(self, val, count=1):
        self.total += val
        self.count += count

    def reset(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.

    @property
    def val(self):
        return self.total / self.count

class EarlyStop(Exception):
    pass

class NLIExperiment(Experiment):
    def __init__(self, args, dataloaders):
        super(NLIExperiment, self).__init__(args, dataloaders)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.updates = 0

        # stats
        self.train_stats = defaultdict(list)
        self.valid_stats = defaultdict(list)
        self._best_acc = 0.

        # create output path
        self.setup()
        self.build_model() # self.model
        self.build_optimizer() #self.optimizer
        self.build_scheduler() #self.scheduler
        self.build_loss_func() #self.loss_func

    def update_stats(self, stats, set_name):
        """
        stats: Dict, 
        """
        if set_name == "train":
            for key, val in stats.items():
                self.train_stats[key] += val 
        elif set_name == "valid":
            for key, val in stats.items():
                self.valid_stats[key] += val
        else:
            raise ValueError(f"{set_name} is not predefined")
        
    def write_stats(self, set_name):
        fn = os.path.join(self.out_dir, "stats_{}.log.gz".format(set_name))
        if set_name == "train":
            with gzip.open(fn, "wt") as fzip:
                json.dump(self.train_stats, fzip) # see https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file
        elif set_name == "valid":
            with gzip.open(fn, "wt") as fzip:
                json.dump(self.valid_stats, fzip)
        else:
            raise ValueError(f"{set_name} is not predefined")

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.verbose:
            self.print_write_to_log(re.sub(r"\n", "", self.optimizer.__repr__()))

    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=self.args.lr_decay,
                                    patience=self.args.decay_patience)
    
    def build_model(self):
        print("Start creating model ...")
        # very dirty writing
        pretrained_embeddings = np.load(self.args.pretrained_path)
        setattr(self.args, "vocab_size", pretrained_embeddings.shape[0])

        if self.args.model == "esim":
            from esim.model import ESIM
            self.model = ESIM(vocab_size=self.args.vocab_size, ebd_dim=self.args.embedding_dim, hidden_dim=self.args.hidden_dim,
                                device=self.device, pretrained_embeddings=pretrained_embeddings, 
                                padding_idx=0, dropout=self.args.dropout, num_classes=self.args.num_classes,
                                freeze_embeddings=self.args.freeze_embeddings)
        elif self.args.model == "re2":
            from re2.model import RE2
            self.model = RE2(vocab_size=self.args.vocab_size, embedding_dim=self.args.embedding_dim, hidden_dim=self.args.hidden_dim, 
                            device=self.device,num_layers=self.args.num_layers, kernel_sizes=self.args.kernel_sizes, 
                            num_blocks=self.args.num_blocks, num_feat_type=4, num_classes=self.args.num_classes, dropout=self.args.dropout, 
                            pretrained_embeddings=pretrained_embeddings, freeze_embeddings=self.args.freeze_embeddings)
        elif self.args.model == "cafe":
            from cafe.model import CAFE
            self.model = CAFE(vocab_size=self.args.vocab_size, embedding_dim=self.args.embedding_dim,hidden_dim=self.args.hidden_dim,
                k_factor=self.args.k_factor, enhance_mode=self.args.enhance_mode, pool_mode=self.args.pool_mode, device=self.device,
                pretrained_embeddings=pretrained_embeddings, freeze_word_embeddings=self.args.freeze_word_embeddings, 
                use_char=self.args.use_char, use_pos=self.args.use_pos, use_local_feat=self.args.use_local_feat, 
                char_dim=self.args.char_dim, pos_dim=self.args.pos_dim, local_feat_dim=self.args.local_feat_dim, 
                char_size=self.args.char_size, pos_size=self.args.pos_size, local_feat_size=self.args.local_feat_size, 
                padding_idx=0, char_kernel_size=self.args.char_kernel_size, 
                dropout=self.args.dropout, word_dropout=self.args.word_dropout, num_classes=3, num_layers=self.args.num_layers)
        else:
            raise ValueError(f"the model {self.args.model} not implemented")
        if self.args.use_pretrain:   
            pass
        else:
            self.print_write_to_log("[Warning]: Not use pretrained embeddings")

        self.model.to(self.device)

    def build_loss_func(self):
        self.loss_func = nn.CrossEntropyLoss()  

    def train_one_epoch(self, current_epoch):
        if self.args.model == "esim" or self.model == "re2":
            self.standard_train_one_epoch(current_epoch)
        elif self.args.model == "cafe":
            self.cafe_train_one_epoch(current_epoch)
        else:
            raise ValueError(f"the model {self.args.model} not implemented")
    
    def valid_one_epoch(self):
        if self.args.model == "esim" or self.model == "re2":
            self.standard_valid_one_epoch()
        elif self.args.model == "cafe":
            self.cafe_valid_one_epoch()
        else:
            raise ValueError(f"the model {self.args.model} not implemented")
    
    def standard_train_one_epoch(self, current_epoch):
        loss_val = 0.
        acc_val = 0.
        avg_loss = AvgMeters()
        avg_acc = AvgMeters()
        start_time = time.time()
        stats = {"acc": [],
                "loss": [],
                "step": []}
        
        self.model.train()
        for i, (prems, prems_lengths, hypos, hypos_lengths, labels) in enumerate(self.train_dataloader):
            self.updates += 1
            prems = prems.to(self.device)
            prems_lengths = prems_lengths.to(self.device)
            hypos = hypos.to(self.device)
            hypos_lengths = hypos_lengths.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred_logits = self.model(prems, hypos, prems_lengths, hypos_lengths)
            loss = self.loss_func(pred_logits, labels)
            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            # metric
            loss_val = loss.mean().item()
            acc_val = correct_instance_count(pred_logits, labels) / len(labels)
            avg_loss.update(loss_val)
            avg_acc.update(acc_val)

            # stats
            if ((i+1) % self.args.stats_idx == 0) and (self.args.stats):
                stats["loss"].append(loss_val)
                stats["acc"].append(acc_val)
                stats["step"].append(self.updates)
                self.update_stats(stats, "train")
            
            # output
            if (i+1) % self.args.log_idx == 0 and self.args.log:
                elpased_time = (time.time() - start_time) / self.args.log_idx

                log_text = "epoch: {}/{}, step: {}/{}, loss: {:.3f}, acc: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                    current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_loss.val, avg_acc.val, 
                    self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                )
                self.print_write_to_log(log_text)

                avg_loss.reset()
                avg_acc.reset() 
                start_time = time.time()

        return stats

    def standard_valid_one_epoch(self):
        self.print_write_to_log("="*50 + "starting vaidation" + "="*50)

        loss_val = 0.
        acc_val = 0.
        correct_count = 0.
        stats = {"acc": [],
                "loss": [],
                "step": []}

        self.model.eval()
        for i, (prems, prems_lengths, hypos, hypos_lengths, labels) in enumerate(self.valid_dataloader):
            prems = prems.to(self.device)
            prems_lengths = prems_lengths.to(self.device)
            hypos = hypos.to(self.device)
            hypos_lengths = hypos_lengths.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                pred_logits = self.model(prems, hypos, prems_lengths, hypos_lengths)
                loss = self.loss_func(pred_logits, labels)
                correct_count += correct_instance_count(pred_logits, labels)
                
                loss_val += loss.item()
            
        loss_val = loss_val / len(self.valid_dataloader)
        acc_val = correct_count / len(self.valid_dataloader.dataset)
        
        # scheduler
        self.scheduler.step(acc_val)

        # output
        if acc_val > self.best_acc:
                self.best_acc = acc_val
                self.save("best_model")
                self.patience = 0
        else:
            self.patience += 1

        log_text = "valid loss: {:.3f}, valid acc: {:.3f}, best acc: {:.3f}".format(loss_val, acc_val*100, self.best_acc*100)
        self.print_write_to_log(log_text)

        # stats
        stats["acc"].append(acc_val)
        stats["loss"].append(loss_val)
        stats["step"].append(self.updates)
        self.update_stats(stats, "valid")

        # ealry stop
        if self.patience >= self.args.patience:
            # write stats 
            if self.args.stats:
                self.write_stats("train")
                self.write_stats("valid")

            raise EarlyStop("early stop")
    
    def cafe_train_one_epoch(self, current_epoch):
        loss_val = 0.
        acc_val = 0.
        avg_loss = AvgMeters()
        avg_acc = AvgMeters()
        start_time = time.time()
        stats = {"acc": [],
                "loss": [],
                "step": []}
        
        self.model.train()
        for i, (prems, prems_lengths, hypos, hypos_lengths, labels, p_chars, q_chars, p_pos, q_pos) in enumerate(self.train_dataloader):
            self.updates += 1
            prems = prems.to(self.device)
            prems_lengths = prems_lengths.to(self.device)
            hypos = hypos.to(self.device)
            hypos_lengths = hypos_lengths.to(self.device)
            labels = labels.to(self.device)
            p_chars = p_chars.to(self.device)
            q_chars = q_chars.to(self.device)
            p_pos = p_pos.to(self.device)
            q_pos = q_pos.to(self.device)

            self.optimizer.zero_grad()
            pred_logits = self.model(prems, hypos, prems_lengths, hypos_lengths, p_chars, q_chars, p_pos, q_pos)
            loss = self.loss_func(pred_logits, labels)
            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            
            # metric
            loss_val = loss.mean().item()
            acc_val = correct_instance_count(pred_logits, labels) / len(labels)
            avg_loss.update(loss_val)
            avg_acc.update(acc_val)

            # stats
            if ((i+1) % self.args.stats_idx == 0) and (self.args.stats):
                stats["loss"].append(loss_val)
                stats["acc"].append(acc_val)
                stats["step"].append(self.updates)
                self.update_stats(stats, "train")
            
            # output
            if (i+1) % self.args.log_idx == 0 and self.args.log:
                elpased_time = (time.time() - start_time) / self.args.log_idx

                log_text = "epoch: {}/{}, step: {}/{}, loss: {:.3f}, acc: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                    current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_loss.val, avg_acc.val, 
                    self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                )
                self.print_write_to_log(log_text)

                avg_loss.reset()
                avg_acc.reset() 
                start_time = time.time()

        return stats

    def cafe_valid_one_epoch(self):
        self.print_write_to_log("="*50 + "starting vaidation" + "="*50)

        loss_val = 0.
        acc_val = 0.
        correct_count = 0.
        stats = {"acc": [],
                "loss": [],
                "step": []}

        self.model.eval()
        for i, (prems, prems_lengths, hypos, hypos_lengths, labels, p_chars, q_chars, p_pos, q_pos) in enumerate(self.valid_dataloader):
            prems = prems.to(self.device)
            prems_lengths = prems_lengths.to(self.device)
            hypos = hypos.to(self.device)
            hypos_lengths = hypos_lengths.to(self.device)
            labels = labels.to(self.device)
            p_chars = p_chars.to(self.device)
            q_chars = q_chars.to(self.device)
            p_pos = p_pos.to(self.device)
            q_pos = q_pos.to(self.device)

            with torch.no_grad():
                pred_logits = self.model(prems, hypos, prems_lengths, hypos_lengths, p_chars, q_chars, p_pos, q_pos)
                loss = self.loss_func(pred_logits, labels)
                correct_count += correct_instance_count(pred_logits, labels)
                
                loss_val += loss.item()
            
        loss_val = loss_val / len(self.valid_dataloader)
        acc_val = correct_count / len(self.valid_dataloader.dataset)
        
        # scheduler
        self.scheduler.step(acc_val)

        # output
        if acc_val > self.best_acc:
                self.best_acc = acc_val
                self.save("best_model")
                self.patience = 0
        else:
            self.patience += 1

        log_text = "valid loss: {:.3f}, valid acc: {:.3f}, best acc: {:.3f}".format(loss_val, acc_val*100, self.best_acc*100)
        self.print_write_to_log(log_text)

        # stats
        stats["acc"].append(acc_val)
        stats["loss"].append(loss_val)
        stats["step"].append(self.updates)
        self.update_stats(stats, "valid")

        # ealry stop
        if self.patience >= self.args.patience:
            # write stats 
            if self.args.stats:
                self.write_stats("train")
                self.write_stats("valid")

            raise EarlyStop("early stop")

    @property
    def best_acc(self):
        return self._best_acc
    
    @best_acc.setter
    def best_acc(self, val):
        self._best_acc = val

    def train(self):
        # print parameters settting & model architecture
        self.print_args()
        self.print_model_stats()
        self.patience = 0

        print("start training ...")
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            self.valid_one_epoch()

class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super().__init__()
        self.args = args 
        self.set_name = set_name
        self.path = self.args.data_dir + f"{set_name}_example.pkl"

        with open(self.path, "rb") as f:
            self._examples = pickle.load(f)

    def __len__(self):
        return len(self._examples)
    
    def __getitem__(self, i):
        example = self._examples[i]
        p, q, p_len, q_len, label = example[0], example[1], example[2], example[3], example[-1]

        return p, q, p_len, q_len, label

    @staticmethod
    def collate_fn(batch):
        ps, qs, p_lens, q_lens, labels = zip(*batch)

        # batch-wise clip
        p_lens = LongTensor(p_lens)
        q_lens = LongTensor(q_lens)
        labels = LongTensor(labels)
        
        p_max_len = p_lens.max().item()
        q_max_len = q_lens.max().item()

        new_ps = torch.zeros(size=(len(batch), p_max_len)).long()
        new_qs = torch.zeros(size=(len(batch), q_max_len)).long()

        for i, (p, p_len) in enumerate(zip(ps, p_lens)):
            new_ps[i, :p_len] = LongTensor(p[:p_len])
        for i, (q, q_len) in enumerate(zip(qs, q_lens)):
            new_qs[i, :q_len] = LongTensor(q[:q_len])


        return new_ps, p_lens, new_qs, q_lens, labels

class NLIDatasetWithFeat(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super().__init__()
        self.args = args 
        self.set_name = set_name
        self.path = self.args.data_dir + f"{set_name}_example.pkl"

        with open(self.path, "rb") as f:
            self._examples = pickle.load(f)

    def __len__(self):
        return len(self._examples)
    
    def __getitem__(self, i):
        example = self._examples[i]
        p, q, p_len, q_len, label = example[0], example[1], example[2], example[3], example[-1]
        p_chars, q_chars = example[4], example[5]
        p_pos, q_pos = example[6], example[7]

        return p, q, p_len, q_len, label, p_chars, q_chars, p_pos, q_pos

    @staticmethod
    def collate_fn(batch):
        ps, qs, p_lens, q_lens, labels, p_chars, q_chars, p_pos, q_pos = zip(*batch)

        # batch-wise clip
        p_lens = LongTensor(p_lens)
        q_lens = LongTensor(q_lens)
        labels = LongTensor(labels)
        
        p_max_len = p_lens.max().item()
        q_max_len = q_lens.max().item()
        char_len = p_chars[0].shape[-1] #NOTE: since p_chars is a tuple

        new_ps = torch.zeros(size=(len(batch), p_max_len)).long()
        new_qs = torch.zeros(size=(len(batch), q_max_len)).long()
        new_p_chars = torch.zeros(size=(len(batch), p_max_len, char_len)).long()
        new_q_chars = torch.zeros(size=(len(batch), q_max_len, char_len)).long() 
        new_p_pos = torch.zeros(size=(len(batch), p_max_len)).long()
        new_q_pos = torch.zeros(size=(len(batch), q_max_len)).long()

        for i, (p, pc, pp, p_len) in enumerate(zip(ps, p_chars, p_pos, p_lens)):
            new_ps[i, :p_len] = LongTensor(p[:p_len])
            new_p_chars[i,:p_len, :] = LongTensor(pc[:p_len, :])
            new_p_pos[i,:p_len] = LongTensor(pp[:p_len])
        for i, (q, qc, qp, q_len) in enumerate(zip(qs, q_chars, q_pos, q_lens)):
            new_qs[i, :q_len] = LongTensor(q[:q_len])
            new_q_chars[i, :q_len, :] = LongTensor(qc[:q_len, :])
            new_q_pos[i, :q_len] = LongTensor(qp[:q_len])

        return new_ps, p_lens, new_qs, q_lens, labels, new_p_chars, new_q_chars, new_p_pos, new_q_pos


if __name__ == "__main__":
    # parse sys args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="re2", help="model name", type=str)
    sys_args = parser.parse_args()

    if sys_args.model == "esim":     
        config_file = "./esim/default.json"
        args = parse_args(config_file)
        train_dataset = NLIDataset(args, "train")
        valid_dataset = NLIDataset(args, "valid")
        train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=NLIDataset.collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=NLIDataset.collate_fn)
    elif sys_args.model == "re2":
        config_file = "./re2/default.json"
        args = parse_args(config_file)
        train_dataset = NLIDataset(args, "train")
        valid_dataset = NLIDataset(args, "valid")
        train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=NLIDataset.collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=NLIDataset.collate_fn)
    elif sys_args.model == "cafe":
        config_file = "./cafe/default.json"
        args = parse_args(config_file)
        train_dataset = NLIDatasetWithFeat(args, "train")
        valid_dataset = NLIDatasetWithFeat(args, "valid")
        train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                        collate_fn=NLIDatasetWithFeat.collate_fn)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                                        collate_fn=NLIDatasetWithFeat.collate_fn)
    else:
        raise ValueError(f"the model {sys_args.model} is not implemented")


    print("train_data length {}, train_dataloder length {}, valid_data length {}, valid_dataloder length {}".format(
        len(train_dataset), len(train_dataloder), len(valid_dataset), len(valid_dataloader)
    ))
    setattr(args, "model", sys_args.model)
    

    dataloaders = {"train": train_dataloder, "valid": valid_dataloader, "test": None}
    # ------------- Create experiment object ----------
    experiment = NLIExperiment(args, dataloaders)
    experiment.train()