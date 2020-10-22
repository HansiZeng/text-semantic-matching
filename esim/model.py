import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .layers import CombineEmbedding, HighWayEncoder, InterSimlarity, Seq2SeqEncoder, SoftAlignment, WordEmbedding
from .utils import *

class ESIM(nn.Module):
    def __init__(self, vocab_size, ebd_dim, hidden_dim, device, pretrained_embeddings= None, 
            padding_idx=0, dropout=0.5, num_classes=3, freeze_embeddings=False):
        super(ESIM, self).__init__()
        self.device = device

        # input encoding
        self.word_ebd = WordEmbedding(vocab_size, ebd_dim, pretrained_embeddings=pretrained_embeddings ,freeze_embeddings=freeze_embeddings)

        # encoder
        self.ctx_enc = Seq2SeqEncoder(nn.LSTM, ebd_dim, hidden_dim, dropout=dropout, batch_first=True, bidirectional=True)

        # local inference modeling
        self.inter_sim = InterSimlarity(2*hidden_dim, 2*hidden_dim, "identity")
        self.soft_alignment = SoftAlignment()

        # inference composition
        self.proj = nn.Sequential(
                                nn.Linear(4*2*hidden_dim, hidden_dim),
                                nn.ReLU())
        self.compsition = Seq2SeqEncoder(nn.LSTM, hidden_dim, hidden_dim, dropout=dropout, batch_first=True, bidirectional=True)

        # classification
        self.classification = nn.Sequential(nn.Dropout(p=dropout),
                                            nn.Linear(8*hidden_dim, hidden_dim),
                                            nn.Tanh(),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(hidden_dim, num_classes))
        
        # initialize weights
        #self.init_weights_()

    
    def set_pretrain(self, val):
        self.word_ebd.set_(val)

    def forward(self, prems, hypos, prems_lengths, hypos_lengths):
        """
        Args:
            prems: LongTensor with shape of [bz, seq_len_p]
            hypos: LongTensor with shape of [bz, seq_len_q]
            prems_lengths: LongTensor with shape of [bz]
            hypos_lengths: LongTensor with shape of [bz]

        Returns:
            logits: FloatTensor with shape of [bz, num_classes]
        """
        # create masks 
        prems_mask = get_mask(prems).to(self.device) #[bz, seq_len_p]
        hypos_mask = get_mask(hypos).to(self.device) #[bz, seq_len_q]

        # input encoding
        prems = self.word_ebd(prems)
        hypos = self.word_ebd(hypos)
        
        ctx_prems = self.ctx_enc(prems, prems_lengths) #[bz, seq_len_p, 2*hidden_dim]
        ctx_hypos = self.ctx_enc(hypos, hypos_lengths) #[bz, seq_len_q, 2*hidden_dim]

        # local inference modeling
        similarity_scores = self.inter_sim(ctx_prems, ctx_hypos) #[bz, seq_len_p, seq_len_q]
        prems_to_hypos_soft_similarity = masked_softmax(similarity_scores, hypos_mask)
        hypos_to_prems_soft_similarity = masked_softmax(similarity_scores.permute(0,2,1), prems_mask)
        aligned_prems = torch.bmm(prems_to_hypos_soft_similarity, ctx_hypos) #[bz, seq_len_p, 2*hidden_dim]
        aligned_hypos = torch.bmm(hypos_to_prems_soft_similarity, ctx_prems) #[bz, seq_len_q, 2*hidden_dim]

        enhanced_prems = torch.cat([ctx_prems, aligned_prems, ctx_prems-aligned_prems, ctx_prems*aligned_prems], dim=-1)
        enhanced_hypos = torch.cat([ctx_hypos, aligned_hypos, ctx_hypos-aligned_hypos, ctx_hypos*aligned_hypos], dim=-1)

        # inference composition
        proj_prems = self.proj(enhanced_prems)
        proj_hypos = self.proj(enhanced_hypos)

        compos_prems = self.compsition(proj_prems, prems_lengths) #[bz, seq_len_p, 2*hidden_dim]
        compos_hypos = self.compsition(proj_hypos, hypos_lengths) #[bz, seq_len_q, 2*hidden_dim]

        # prepare for classification input
        compos_prems = compos_prems.masked_fill(~prems_mask.unsqueeze(2), 0.0)
        compos_hypos = compos_hypos.masked_fill(~hypos_mask.unsqueeze(2), 0.0)

        avg_prems = compos_prems.mean(dim=1)
        max_prems, _ = compos_prems.max(dim=1) #[bz, 2*hidden_dim]
        avg_hypos = compos_hypos.mean(dim=1)
        max_hypos, _ = compos_hypos.max(dim=1) #[bz, 2*hidden_dim]

        in_feat = torch.cat([avg_prems, max_prems, avg_hypos, max_hypos], dim=-1)

        # classification
        logits = self.classification(in_feat)

        return logits


class CharESIM(nn.Module):
    def __init__(self, vocab_size, ebd_dim, hidden_dim, device, 
                pretrained_embeddings=None, freeze_word_embeddings=True, use_char=False, 
                use_pos=False, use_local_feat=False, char_dim=None, pos_dim=None, local_feat_dim=None, 
                char_size=None, pos_size=None, local_feat_size=None, padding_idx=0, char_kernel_size=3, 
                dropout=0.2, word_dropout=0.2, num_classes=3, num_layers=2):
        super(CharESIM, self).__init__()
        self.device = device

        # input encoding
        self.embeddinngs = CombineEmbedding(vocab_size, ebd_dim, pretrained_embeddings=pretrained_embeddings,
                            freeze_word_embeddings=freeze_word_embeddings, use_char=use_char, use_pos=use_pos,
                            use_local_feat=use_local_feat, char_dim=char_dim, pos_dim=pos_dim, local_feat_dim=local_feat_dim,
                            char_size=char_size, pos_size=pos_size, local_feat_size=local_feat_size, padding_idx=padding_idx,
                            dropout=word_dropout, char_kernel_size=char_kernel_size)
        in_feat = ebd_dim 
        if use_char:
            in_feat += char_dim
        if use_pos:
            in_feat += pos_dim
        if use_local_feat:
            in_feat += local_feat_dim

        self.highway_encoder = HighWayEncoder(in_feat, hidden_dim, num_layers=num_layers, dropout=dropout) # NOTE: not use dropout !!!

        # encoders
        self.ctx_enc = Seq2SeqEncoder(nn.LSTM, hidden_dim, hidden_dim, dropout=dropout, batch_first=True, bidirectional=True)

        # local inference modeling
        self.inter_sim = InterSimlarity(2*hidden_dim, 2*hidden_dim, "identity")
        self.soft_alignment = SoftAlignment()

        # inference composition
        self.proj = nn.Sequential(
                                nn.Linear(4*2*hidden_dim, hidden_dim),
                                nn.ReLU())
        self.compsition = Seq2SeqEncoder(nn.LSTM, hidden_dim, hidden_dim, dropout=dropout, batch_first=True, bidirectional=True)

        # classification
        self.classification = nn.Sequential(nn.Dropout(p=dropout),
                                            nn.Linear(8*hidden_dim, hidden_dim),
                                            nn.Tanh(),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(hidden_dim, num_classes))
        
        # initialize weights
        #self.init_weights_()

    
    def forward(self, prems, hypos, prems_lengths, hypos_lengths, p_chars, q_chars, p_pos, q_pos):
        """
        Args:
            prems: LongTensor with shape of [bz, seq_len_p]
            hypos: LongTensor with shape of [bz, seq_len_q]
            prems_lengths: LongTensor with shape of [bz]
            hypos_lengths: LongTensor with shape of [bz]
            p_chars: [bz, seq_len_p, char_max]
            p_pos: [bz, seq_len_p]

        Returns:
            logits: FloatTensor with shape of [bz, num_classes]
        """
        # create masks 
        prems_mask = get_mask(prems).to(self.device) #[bz, seq_len_p]
        hypos_mask = get_mask(hypos).to(self.device) #[bz, seq_len_q]

        # input encoding
        prems = self.embeddinngs(prems, char_inputs=p_chars, pos_inputs=p_pos)
        hypos = self.embeddinngs(hypos, char_inputs=q_chars, pos_inputs=q_pos)
        prems = self.highway_encoder(prems)
        hypos = self.highway_encoder(hypos)
        
        ctx_prems = self.ctx_enc(prems, prems_lengths) #[bz, seq_len_p, 2*hidden_dim]
        ctx_hypos = self.ctx_enc(hypos, hypos_lengths) #[bz, seq_len_q, 2*hidden_dim]

        # local inference modeling
        similarity_scores = self.inter_sim(ctx_prems, ctx_hypos) #[bz, seq_len_p, seq_len_q]
        prems_to_hypos_soft_similarity = masked_softmax(similarity_scores, hypos_mask)
        hypos_to_prems_soft_similarity = masked_softmax(similarity_scores.permute(0,2,1), prems_mask)
        aligned_prems = torch.bmm(prems_to_hypos_soft_similarity, ctx_hypos) #[bz, seq_len_p, 2*hidden_dim]
        aligned_hypos = torch.bmm(hypos_to_prems_soft_similarity, ctx_prems) #[bz, seq_len_q, 2*hidden_dim]

        enhanced_prems = torch.cat([ctx_prems, aligned_prems, ctx_prems-aligned_prems, ctx_prems*aligned_prems], dim=-1)
        enhanced_hypos = torch.cat([ctx_hypos, aligned_hypos, ctx_hypos-aligned_hypos, ctx_hypos*aligned_hypos], dim=-1)

        # inference composition
        proj_prems = self.proj(enhanced_prems)
        proj_hypos = self.proj(enhanced_hypos)

        compos_prems = self.compsition(proj_prems, prems_lengths) #[bz, seq_len_p, 2*hidden_dim]
        compos_hypos = self.compsition(proj_hypos, hypos_lengths) #[bz, seq_len_q, 2*hidden_dim]

        # prepare for classification input
        compos_prems = compos_prems.masked_fill(~prems_mask.unsqueeze(2), 0.0)
        compos_hypos = compos_hypos.masked_fill(~hypos_mask.unsqueeze(2), 0.0)

        avg_prems = compos_prems.mean(dim=1)
        max_prems, _ = compos_prems.max(dim=1) #[bz, 2*hidden_dim]
        avg_hypos = compos_hypos.mean(dim=1)
        max_hypos, _ = compos_hypos.max(dim=1) #[bz, 2*hidden_dim]

        in_feat = torch.cat([avg_prems, max_prems, avg_hypos, max_hypos], dim=-1)

        # classification
        logits = self.classification(in_feat)

        return logits



if __name__ == "__main__":
    sample_prems = torch.LongTensor([[9,4,5,2,0], [3,2,1,0,0], [2,1,4,5,5]])
    sample_hypos = torch.LongTensor([[8,2,4,2], [1,2,0,0], [6,3,4,0]])
    sample_prems_lengths = (sample_prems != 0).sum(-1)
    sample_hypos_lengths = (sample_hypos != 0).sum(-1)

    esim = ESIM(10, 8, 8, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    out_logits = esim(sample_prems, sample_hypos, sample_prems_lengths, sample_hypos_lengths)
    print(out_logits)

    for module in list(esim.modules()):
        if isinstance(module, nn.Sequential):
            print(module[0])
