import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *


class VariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):

        """
        Apply dropout to input tensor.
        # Parameters
        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`
        # Returns
        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, padding_idx=0, freeze_embeddings=False):
        super(WordEmbedding, self).__init__()

        self.freeze_embeddings = freeze_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.requires_grad = not self.freeze_embeddings
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(pretrained_embeddings)})
        else:
            print("[Warning] not use pretrained embeddings ...")

    def forward(self, inputs):
        out = self.embedding(inputs)     
        return out

class CharEmbedding(nn.Module):
    """
    v1: easiest version
    """
    def __init__(self, vocab_size, embedding_dim, kernel_size, dropout=0., padding_idx=0, freeze_embeddings=False):
        super(CharEmbedding, self).__init__()

        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embeddings.weight.requires_grad = not freeze_embeddings

        self.encoder = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
                                    

    def forward(self, inputs):
        """
        inputs: [bz, seq_len, char_max]
        """
        bz, seq_len, char_max = list(inputs.size())
        inputs = self.embeddings(inputs).view(bz*seq_len, char_max, -1)
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        inputs = inputs.transpose(1,2) #[bz*seq_len, out_feat, char_max]
        outputs = self.encoder(inputs)
        outputs = F.max_pool1d(outputs, char_max).squeeze(2) #[bz*seq_len, out_feat]
        outputs = outputs.view(bz, seq_len, self.embedding_dim)

        return outputs

class FeatEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0., padding_idx=0, freeze_embeddings=False):
        super(FeatEmbedding, self).__init__()
        self.freeze_embeddings = freeze_embeddings
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embeddings.weight.requires_grad = not self.freeze_embeddings

    def forward(self, inputs):
        """
        Args:
            inputs: [bz, seq_len]
        """

        outputs = self.embeddings(inputs) #[bz, seq_len, embedding_dim]

        return outputs

class CombineEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, freeze_word_embeddings=True, use_char=False, 
            use_pos=False, use_local_feat=False, char_dim=None, pos_dim=None, local_feat_dim=None, 
            char_size=None, pos_size=None, local_feat_size=None, dropout=0., padding_idx=0, char_kernel_size=3):
        super(CombineEmbedding, self).__init__()
        self.use_char = use_char
        self.use_pos = use_pos
        self.use_local_feat = use_local_feat

        self.word_embedddings = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                        freeze_embeddings=freeze_word_embeddings)

        if use_char:
            self.char_embeddings = CharEmbedding(char_size, char_dim, char_kernel_size)
        if use_pos:
            self.pos_embeddings = FeatEmbedding(pos_size, pos_dim)
        if use_local_feat:
            self.local_feat_embeddings = FeatEmbedding(local_feat_size, local_feat_dim)

        # NOTE: apply dropout after concatenation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
    def forward(self, word_inputs, char_inputs=None, pos_inputs=None, local_feat_inputs=None):
        word_inputs = self.word_embedddings(word_inputs)
        outputs = word_inputs

        if self.use_char:
            char_inputs = self.char_embeddings(char_inputs)
            outputs = torch.cat([outputs, char_inputs], dim=-1)
        if self.use_pos:
            pos_inputs = self.pos_embeddings(pos_inputs)
            outputs = torch.cat([outputs, pos_inputs], dim=-1)
        if self.use_local_feat:
            local_feat_inputs = self.local_feat_embeddings(local_feat_inputs)
            outputs = torch.cat([outputs, local_feat_inputs], dim=-1)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs

class HighwayLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation=nn.ReLU, dropout=0.):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.trans_layer = nn.Sequential(nn.Linear(in_feat, out_feat),
                                        activation())
        self.gate_layer = nn.Sequential(nn.Linear(in_feat, out_feat),
                                        nn.Sigmoid())

        self.projection = True if in_feat != out_feat else False
        if self.projection:
            self.proj_layer = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, tensor):
        trans_tensor = self.trans_layer(tensor)
        gate_tensor = self.gate_layer(tensor)

        if self.projection:
            out_tensor = trans_tensor * gate_tensor + (1-gate_tensor) * self.proj_layer(tensor)
        else:
            out_tensor = trans_tensor * gate_tensor + (1-gate_tensor) * tensor 

        if self.dropout is not None:
            out_tensor = self.dropout(out_tensor)

        return out_tensor

class HighWayEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, num_layers, activation=nn.ReLU, dropout=0.):
        """
        NOTE: Apply `dropout` on each output of highlayer
        """
        super(HighWayEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HighwayLayer(in_feat, out_feat, activation=activation, dropout=dropout))
            elif i == num_layers-1:
                # dropout = 0., since the model applys variational dropout before the LSTM encoder (the output of the HighwayEncoder)
                layers.append(HighwayLayer(out_feat, out_feat, activation=activation, dropout=0.))
            else:
                layers.append(HighwayLayer(out_feat, out_feat, activation=activation, dropout=dropout))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, inputs):
        """
        NOTE: Should I apply mask?
        """
        outputs = self.layers(inputs)
        return outputs

class Seq2SeqEncoder(nn.Module):
    """
    Wrapper of the torch.nn.RNNBase module. This module can handle the batch of variable lengths
    padded sequences.
    """
    def __init__(self, rnn_type, input_dim, hidden_dim, dropout=0., batch_first=True, bidirectional=False):
        super(Seq2SeqEncoder, self).__init__()

        if dropout:
            self.dropout = VariationalDropout(p=dropout)
        else:
            self.dropout = None

        self.rnn = rnn_type(input_dim, hidden_dim, batch_first=batch_first, bidirectional=bidirectional)
        

    def forward(self, seq, seq_lengths):
        """
        Args:
            seq: FloatTensor with shape of [bz, max_seq_len, input_dim]
            seq_lengths: LongTensor with shape of [bz]

        Returns:
            out: FloatTensor with shape of [bz, max_seq_len, hidden_dim]
        """
        if self.dropout is not None:
            seq = self.dropout(seq)

        packed_seq = pack_padded_sequence(seq, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed_seq, None)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return out


class InterSimlarity(nn.Module):
    """
    Compute the inter similarity score between two sequences
    """
    def __init__(self, in_feat, out_feat, type):
        super(InterSimlarity, self).__init__()

        self.fn = None
        if type == "identity":
            pass 
        elif type == "1-layer-feed-forward-relu":
            self.fn = nn.Sequential(nn.Linear(in_feat, out_feat),
                                    nn.ReLU())
        else:
            raise ValueError(f"type {type} is not predefined")

    def forward(self, seq_prev, seq_after):
        """
        Args:
            seq_prev: FloatTensor with shape of [bz, seq_len_1, in_feat]
            seq_after: FloatTensor with shape of [bz, seq_len_2, in_feat]
           
        Returns:
            sim_score: FloatTensor with shape of [bz, seq_len_1, seq_len_2]
        """
        if self.fn:
            seq_prev = self.fn(seq_prev)
            seq_after = self.fn(seq_after)
        
        seq_prev = seq_prev.unsqueeze(2) # [bz, seq_len_prev, 1, in_feat]
        seq_after = seq_after.unsqueeze(1) # [bz, 1, seq_len_after, in_feat]
        
        sim_score = torch.mul(seq_prev, seq_after).sum(-1) #[bz, seq_len_prev, seq_len_after]

        return sim_score

class SoftAlignment(nn.Module):
    def forward(self, soft_similarity, context_sequences):
        """
        Args:
            soft_similarity: FloatTensor with shape of [bz, seq_len_aligned, seq_len_context]. It is already masked.
            context_sequences: FloatTensor with shape of [bz, seq_len_context, hidden_dim]
        
        Returns:
            aligned_sequences: FloatTensor with shape of [bz, seq_len_aligned, hidden_dim]
        """
        aligned_sequences = torch.bmm(soft_similarity, context_sequences)
        return aligned_sequences