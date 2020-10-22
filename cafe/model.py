import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .layers import  CombineEmbedding,  HighWayEncoder, HighwayLayer, CoAttention, EnhancedFeature, Seq2SeqEncoder, SequentialPooling
from .utils import get_mask

class CAFE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, k_factor, enhance_mode, pool_mode, device,
                pretrained_embeddings=None, freeze_word_embeddings=True, use_char=False, 
                use_pos=False, use_local_feat=False, char_dim=None, pos_dim=None, local_feat_dim=None, 
                char_size=None, pos_size=None, local_feat_size=None, padding_idx=0, char_kernel_size=3, 
                dropout=0.2, word_dropout=0.2, num_classes=3, num_layers=2):
        super(CAFE, self).__init__()

        self.device = device

        self.embeddings = CombineEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                            freeze_word_embeddings=freeze_word_embeddings, use_char=use_char, use_pos=use_pos,
                            use_local_feat=use_local_feat, char_dim=char_dim, pos_dim=pos_dim, local_feat_dim=local_feat_dim,
                            char_size=char_size, pos_size=pos_size, local_feat_size=local_feat_size, padding_idx=padding_idx,
                            dropout=word_dropout, char_kernel_size=char_kernel_size)
        in_feat = embedding_dim
        if use_char:
            in_feat += char_dim
        if use_pos:
            in_feat += pos_dim
        if use_local_feat:
            in_feat += local_feat_dim
        self.highway_encoder = HighWayEncoder(in_feat, hidden_dim, num_layers=num_layers, dropout=dropout) # NOTE: not use dropout !!!

        self.inter_alignment = CoAttention(hidden_dim, hidden_dim)
        self.intra_alignment = CoAttention(hidden_dim, hidden_dim)

        self.inter_enhancement = EnhancedFeature(hidden_dim, k_factor=k_factor, mode=enhance_mode)
        self.intra_enhancement = EnhancedFeature(hidden_dim, k_factor=k_factor, mode=enhance_mode)

        extra_dim = len(enhance_mode.split("_"))
        print("number of extra feature for enhancement: {}, are {}".format(extra_dim, enhance_mode.split("_")))
        self.rnn = Seq2SeqEncoder(nn.LSTM, 2*extra_dim+hidden_dim, hidden_dim, batch_first=True, bidirectional=False, rnn_dropout=dropout)

        self.sequential_pool = SequentialPooling(pool_mode)

        in_feat_count = len(pool_mode.split("_"))
        self.prediction = nn.Sequential(HighWayEncoder(hidden_dim*in_feat_count*4, hidden_dim, num_layers=num_layers, dropout=dropout),
                                        nn.Linear(hidden_dim, num_classes))

    def num_parameters(self):
        num_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        return num_parameters

    def forward(self, premise, hypothesis, premise_lengths, hypothesis_lenghts, p_chars, q_chars, p_pos, q_pos):
        premise_mask = get_mask(premise).to(self.device)
        hypothesis_mask = get_mask(hypothesis).to(self.device)

        premise = self.embeddings(premise, p_chars, p_pos)
        hypothesis = self.embeddings(hypothesis, q_chars, q_pos)
        premise = self.highway_encoder(premise)
        hypothesis = self.highway_encoder(hypothesis)

        inter_premise, inter_hypothesis, _, _, _ = self.inter_alignment(premise, hypothesis, premise_mask, hypothesis_mask)
        intra_premise, _, _, _, _ = self.intra_alignment(premise, premise, premise_mask, premise_mask)
        intra_hypothesis, _, _, _, _ = self.intra_alignment(hypothesis, hypothesis, hypothesis_mask, hypothesis_mask)

        inter_enhance_premise, inter_enhance_hypothesis = self.inter_enhancement(premise, hypothesis,
                                                                                inter_premise, inter_hypothesis,
                                                                                 premise_mask, hypothesis_mask)
        intra_enhance_premise, intra_enhance_hypothesis = self.intra_enhancement(premise, hypothesis,
                                                                                intra_premise, intra_hypothesis,
                                                                                premise_mask, hypothesis_mask)
        enhance_premise = torch.cat([premise, inter_enhance_premise, intra_enhance_premise], dim=-1)
        enhance_hypothesis = torch.cat([hypothesis, inter_enhance_hypothesis, intra_enhance_hypothesis], dim=-1)

        compose_premise = self.rnn(enhance_premise, premise_lengths)
        compose_hypothesis = self.rnn(enhance_hypothesis, hypothesis_lenghts)

        pool_premise = self.sequential_pool(compose_premise)
        pool_hypothesis = self.sequential_pool(compose_hypothesis)

        input_feature = torch.cat([pool_premise, pool_hypothesis, pool_premise*pool_hypothesis, pool_premise-pool_hypothesis], dim=-1)

        out_logits = self.prediction(input_feature)

        return out_logits



if __name__ == "__main__":
    sample_prems = torch.LongTensor([[9,4,5,2,0], [3,2,1,0,0], [2,1,4,5,5]])
    sample_hypos = torch.LongTensor([[8,2,4,2], [1,2,0,0], [6,3,4,0]])
    sample_prems_lengths = (sample_prems != 0).sum(-1)
    sample_hypos_lengths = (sample_hypos != 0).sum(-1)

    cafe = CAFE(10, 50, 25, 8, "MUL_MIN_CAT", "MAX_AVG", 0., 3)
    out_logits = cafe(sample_prems, sample_hypos, sample_prems_lengths, sample_hypos_lengths)
    print(out_logits)

