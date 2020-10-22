import torch.nn as nn
import torch 

from .layers import WordEmbedding, Encoder, MaskedIdentityAlignment, FullFusion, AugmentedConnection, MaskedPooling, Prediction 
from .utils import get_mask




class RE2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device, num_layers, kernel_sizes, num_blocks, 
                num_feat_type, num_classes, dropout, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        self.device = device
        #self.word_embeddings = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, fix_embeddings=fix_embeddings, dropout=dropout)
        self.word_ebd = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings ,freeze_embeddings=freeze_embeddings)

        self.blocks = nn.ModuleList([nn.ModuleDict({
            "encoder": Encoder(num_layers=num_layers, in_feat=embedding_dim if i==0 else embedding_dim+hidden_dim, out_feat=hidden_dim,
                                kernel_sizes=kernel_sizes, dropout=dropout),
            "alignment": MaskedIdentityAlignment(hidden_dim=embedding_dim+hidden_dim if i==0 else embedding_dim + hidden_dim*2),
            "fusion": FullFusion(in_feat=embedding_dim+hidden_dim if i==0 else embedding_dim + hidden_dim*2, dropout=dropout, 
                                out_feat=hidden_dim)
        }) for i in range(num_blocks)])

        self.connection = AugmentedConnection()
        self.pooling = MaskedPooling()
        self.prediction = Prediction(hidden_dim=hidden_dim, num_feat_type=num_feat_type, dropout=dropout, num_classes=num_classes)
   
    def forward(self, premise, hypothesis, prems_lengths=None, hypos_lengths=None):
        """
        Args: 
            premise: LongTensor with shape of [bz, seq_len_p]
            hypothesis: LongTensor with shape of [bz, seq_len_q]

        Returns:
            out_logits: FloatTensor with shape of [bz, num_classes]
        """
        premise_mask, hypothesis_mask = get_mask(premise).to(self.device), get_mask(hypothesis).to(self.device)
        
        #premise, hypothesis = self.word_embeddings(premise), self.word_embeddings(hypothesis)
        premise = self.word_ebd(premise)
        hypothesis = self.word_ebd(hypothesis)

        prev_premise, prev_hypothesis = premise, hypothesis

        for i, block in enumerate(self.blocks):
            if i > 0:
                premise = self.connection(premise, prev_premise, i)
                hypothesis = self.connection(hypothesis, prev_hypothesis, i)
                prev_premise, prev_hypothesis = premise, hypothesis
            
            
            enc_premise= block["encoder"](premise, premise_mask)
            enc_hypothesis = block["encoder"](hypothesis, hypothesis_mask)

            premise = torch.cat([enc_premise, premise], dim=-1)
            hypothesis = torch.cat([enc_hypothesis, hypothesis], dim=-1)    
            aligned_premise, aligned_hypothesis = block["alignment"](premise, hypothesis, premise_mask, hypothesis_mask)


            fuse_premise = block["fusion"](premise, aligned_premise)
            fuse_hypothesis = block["fusion"](hypothesis, aligned_hypothesis)

            premise, hypothesis = fuse_premise, fuse_hypothesis

        premise = self.pooling(premise, premise_mask)
        hypothesis = self.pooling(hypothesis, hypothesis_mask)

        return self.prediction(premise, hypothesis)


if __name__ == "__main__":
    premise = torch.randint(0, 100, [8, 25])
    hypothesis = torch.randint(0, 100, [8, 18])

    model = RE2(100, 300, 150, 1, [3,5], 1, 4, 3, 0.2)
    out_logits = model(premise, hypothesis)
    print(out_logits, out_logits.mean(), out_logits.std())
    print(out_logits.shape)