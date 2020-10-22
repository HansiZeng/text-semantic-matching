import math

import torch 
import torch.nn as nn
import torch.nn.functional as F

# conv dropout
# 

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
        
class ReluConv1d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_sizes, dropout):
        """
        Features: 
                  - support NxLxC format
                  - allow use different kernel size (should be odd number)
                  - apply dropout before convolution
                  - using Gelu as activation
                  - using kaiming_initialization 
                  - using weight_norm

        Args:
            in_feat: int
            out_feat: int
            kernel_sizes: List of int
            dropout_rate: float 
        """
        super().__init__()
        assert all(kz % 2 == 1 for kz in kernel_sizes) # require each kernel size as odd number 
        assert out_feat % len(kernel_sizes) == 0 # requere the number of different size of kernel should be divisble by out_feat 

        convs = []
        for kz in kernel_sizes:
            conv = nn.Conv1d(in_feat, out_feat//len(kernel_sizes), kz, padding=(kz-1)//2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (kz* in_feat)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(
                nn.utils.weight_norm(conv),
                nn.GELU(),
                nn.Dropout(p=dropout)
                ))
        
        self.encoders = nn.ModuleList(convs)

    def forward(self, tensor):
        """
        Args:
            tensor: FloatTensor with shape of [bz, seq_len, in_feat]

        Returns:
            out: FloatTensor with shape of [bz, seq_len, out_feat]
        """
        tensor = tensor.transpose(2,1)
        out = torch.cat([encoder(tensor) for encoder in self.encoders], dim=1)
        out = out.transpose(2,1)

        return out 

class ReluLinear(nn.Module):
    def __init__(self, in_feat, out_feat, dropout, use_activation=False):
        """
        Feature: 
                 - apply dropout before the layer
                 - use Gelu as activation
                 - use kaiming_initialization 
                 - use weight_norm

        Args:
            pass
        
        Returns:
            out: FloatTensor with shape of [bz, out_feat]
        """
        super().__init__()
        chains = []
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
            chains.append(self.dropout)

        linear = nn.Linear(in_feat, out_feat)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if use_activation else 1.) / in_feat))
        nn.init.zeros_(linear.bias)
        chains.append(nn.utils.weight_norm(linear)) # use weight norm

        if use_activation:
            chains.append(nn.GELU())

        self.chains = nn.Sequential(*chains)

    def forward(self, tensor):
        """
        Args:
            tensor: FloatTensor with shape of [bz, *, in_feat]
        
        Returns:
            tensor: FloatTensor with shape of [bz, *, out_feat]
        """
        return self.chains(tensor)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fix_embeddings, dropout):
        super().__init__()

        self.fix_embeddings = fix_embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
    
    def set_(self, val):
        self.embedding.weight.data = torch.Tensor(val)
        self.embedding.weight.require_grad = not self.fix_embeddings
    
    def forward(self, indices):
        """
        Args: 
            indices: LongTensor with shape of [bz, *]
        Returns:
            out: FloatTensor with shape of [bz, *, embedding_dim]
        """
        out = self.embedding(indices)
        out = self.dropout(out)

        return out

class Encoder(nn.Module):
    def __init__(self, num_layers, in_feat, out_feat, kernel_sizes, dropout):
        super().__init__()
        self.encoders = nn.ModuleList([
            ReluConv1d(in_feat if i==0 else out_feat, out_feat, kernel_sizes, dropout) for i in range(num_layers)
        ])
    
    def forward(self, tensor, mask):
        """
        Args:
            tensor: [bz, seq_len, dim]
            mask: [bz, seq_len]
        
        Returns:
            out: [bz, seq_len, dim]
        """
        if mask.dim() < tensor.dim():
            mask = mask.unsqueeze(-1)

        for encoder in self.encoders:
            tensor = encoder(tensor)
            tensor = tensor.masked_fill(~mask, 0.)

        return tensor 
    
class MaskedIdentityAlignment(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.inverse_temperature = nn.Parameter(torch.tensor(1. / math.sqrt(hidden_dim))) # set temperature as a parameter. BUT DONT WHY.
    
    def forward(self, sequence_a, sequence_b, mask_a, mask_b):
        """
        Args:
            sequence_a: FloatTensor with shape of [bz, seq_a, dim]
            sequence_b: FloatTensor with shape of [bz, seq_b, dim]
            mask_a: BoolTensor with shape of [bz, seq_a]
            mask_b: BoolTensor with shape of [bz, seq_b]
        
        Returns:
            alignment_a: FloatTensor with shape of [bz, seq_a, dim]
            alignment_b: FloatTensor with shape of [bz, seq_b, dim]
        """
        similarity_matrix = torch.bmm(sequence_a, sequence_b.transpose(1,2)) * self.inverse_temperature #[bz, seq_a, seq_b]
        mask_a = mask_a.unsqueeze(1)
        mask_b = mask_b.unsqueeze(1)

        a_to_b_soft_similarity = F.softmax(similarity_matrix.masked_fill(~mask_b, -1e8), dim=-1)
        alignment_a = torch.bmm(a_to_b_soft_similarity, sequence_b)

        b_to_a_soft_similarity = F.softmax(similarity_matrix.transpose(1,2).masked_fill(~mask_a, -1e8), dim=-1)
        alignment_b = torch.bmm(b_to_a_soft_similarity, sequence_a)
        
        return alignment_a, alignment_b
        
class FullFusion(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super().__init__()
        self.concat_fusion = ReluLinear(2*in_feat, out_feat, 0., True)
        self.sub_fusion = ReluLinear(2*in_feat, out_feat, 0., True)
        self.mul_fusion = ReluLinear(2*in_feat, out_feat, 0., True)
        self.aggregate_fusion = ReluLinear(3*out_feat, out_feat, 0., True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, align_tensor):
        """
        Args:
            tensor: FloatTensor with shape of [bz, seq_len, dim]
            align_tensor: FloatTensor with shape of [bz, seq_len, dim]
        
        Returns:
            aggregate_feat: FloatTensor with shape of [bz, seq_len, dim]
        """
        concat_feat = self.concat_fusion(torch.cat([tensor, align_tensor], dim=-1))
        sub_feat = self.sub_fusion(torch.cat([tensor, tensor-align_tensor], dim=-1))
        mul_feat = self.mul_fusion(torch.cat([tensor, tensor*align_tensor], dim=-1))

        aggregate_feat = torch.cat([concat_feat, sub_feat, mul_feat], dim=-1)
        aggregate_feat = self.dropout(aggregate_feat)
        aggregate_feat = self.aggregate_fusion(aggregate_feat)

        return aggregate_feat

class AugmentedConnection(nn.Module):
    """
    Organize the input feature for next block.
    """
    def forward(self, feat, aug_feat, i):
        """
        i == 1: aug_feat: word_embedding
        i >  1: aug_feat: [word_embedding, prev_feat]

        Args:
            feat: [bz, seq_len, hidden_dim]
            aug_feat: [bz, seq_len, embedding_dim] if i == 1
                       [bz, seq_len, embedding_dim+hidden_dim] if i > 1
        
        Returns:
            fuse_feat: [bz, seq_len, embedding_dim+hidden_dim]
        """
        assert i >= 1
        if i == 1:
            fuse_feat = torch.cat([aug_feat, feat], dim=-1)
            return fuse_feat
        else:
            hidden_dim = feat.size()[-1]
            embedding_dim = aug_feat.size()[-1] - hidden_dim

            word_feat = aug_feat[:, :, :embedding_dim]
            feat = (aug_feat[:, :, embedding_dim:] + feat) * math.sqrt(0.5)
            fuse_feat = torch.cat([word_feat, feat], dim=-1)
            return fuse_feat

class MaskedPooling(nn.Module):
    def forward(self, tensor, mask):
        """
        Max pooling over time.
        Args:
            tensor: FloatTensor with shape of [bz, seq_len, dim]
            mask: BoolTensor with shape of[bz, seq_len]

        Returns:
            out: FloatTensor with shape of [bz, dim]
        """
        return tensor.masked_fill(~mask.unsqueeze(-1), -float("inf")).max(dim=1)[0]


class Prediction(nn.Module):
    def __init__(self, hidden_dim, num_feat_type, dropout, num_classes):
        super().__init__()
        self.prediction = nn.Sequential(
                                nn.Dropout(p=dropout),
                                ReluLinear(hidden_dim*num_feat_type, hidden_dim, dropout, True),
                                nn.Dropout(p=dropout),
                                ReluLinear(hidden_dim, num_classes, dropout, False)
                            )
    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: FloatTensor with shape of [bz, dim]
            feat_b: FloatTensor with shape of [bz, dim]
        Returns:
            out_logits: FloatTensor with shape of [bz, num_classes]
        """
        feat = torch.cat([feat_a, feat_b, feat_a-feat_b, feat_a*feat_b], dim=-1)
        out_logits = self.prediction(feat)
        return out_logits

if __name__ == "__main__":
    ### test whether the variance scale can be preserverd when we meet very deep neural network
    # - PROBLEM: the GELU() activation is not compatible with kaiming_uniform with math.sqrt(2 / in_fan)
    print("test linear")
    in_tensor = torch.randn(36, 29, 300)
    deep_linear_model = [ReluLinear(300, 300, 0.2, True) for i in range(100)]
    deep_linear_model = nn.Sequential(*deep_linear_model)
    #deep_linear_model.train()
    deep_linear_model.eval()
    out_tensor = deep_linear_model(in_tensor)
    print("mean {}, std {}".format(out_tensor.mean(), out_tensor.std()))

    print("test convolution")
    deep_conv_encoder = Encoder(100, 300, 300, [3,5], 0.2)
    #deep_conv_model.train()
    deep_conv_encoder.eval()
    out_tensor = deep_conv_encoder(in_tensor)
    print("mean {}, std {}".format(out_tensor.mean(), out_tensor.std()))
    print("------------------------\n")

    ### test whether variance scale can be preserved when stack alignment layers
    seq_a = torch.randn(2,4,3)
    seq_a[0, 2, :], seq_a[0, 3,:] = 0., 0.
    seq_a[1, 3, :] = 0.
    mask_a = ~(seq_a.sum(dim=-1) == 0.)

    seq_b = torch.randn(2,5,3)
    seq_b[0, 3, :], seq_b[0, 4,:] = 0., 0.
    seq_b[1, 4, :] = 0.
    mask_b = ~(seq_b.sum(dim=-1) == 0.)

    alignment_layer = MaskedIdentityAlignment(hidden_dim=3)
    deep_alignment_layers = [Mask]

    ### test MaskedIdentityAlignment
    print("test MaskedIdentityAlignment")
    seq_a = torch.randn(2,4,3)
    seq_a[0, 2, :], seq_a[0, 3,:] = 0., 0.
    seq_a[1, 3, :] = 0.
    mask_a = ~(seq_a.sum(dim=-1) == 0.)

    seq_b = torch.randn(2,5,3)
    seq_b[0, 3, :], seq_b[0, 4,:] = 0., 0.
    seq_b[1, 4, :] = 0.
    mask_b = ~(seq_b.sum(dim=-1) == 0.)

    print("mask_a: {}, mask_b: {}".format(mask_a, mask_b))
    alignment = MaskedIdentityAlignment(seq_a.shape[-1])
    align_a, align_b = alignment(seq_a, seq_b, mask_a, mask_b)
    print("alignment_a: {}|{}, alignment_b: {}|{}".format(align_a, align_a.shape, align_b, align_b.shape))
    print("------------------------\n")
    ### test FullFusion 
    print("test FullFusion Layer")
    full_fusion_layer = FullFusion(seq_a.shape[-1], 0.0)
    fusion_a = full_fusion_layer(seq_a, align_a)
    fusion_b = full_fusion_layer(seq_b, align_b)
    print("fusion_a {}, fusion_b {}".format(fusion_a, fusion_b))
    print("------------------------\n")
    ### test Pooling
    print("test Pooling layer")
    pooling_layer = Pooling()
    feat_a, feat_b = pooling_layer(fusion_a, mask_a), pooling_layer(fusion_b, mask_b)
    print("feat_a {}, feat_b {}".format(feat_a, feat_b))
    print("------------------------\n")

        
