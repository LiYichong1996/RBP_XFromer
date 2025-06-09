import math
from pathlib import Path
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot

from enformer_pytorch.config_enformer import EnformerConfig

from transformers import PreTrainedModel

from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel, BertAttention, BertIntermediate, BertSelfAttention
from torch import Tensor, device, dtype, nn
from typing import Optional, Set, Tuple, Dict, Any

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

# from my_network.rbp_attention import MultiLabelAttention

# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# losses and metrics

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

# relative positional encoding functions

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    if not exists(stddev):
        stddev = seq_len / (2 * features)

    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)#.half()
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        if attn_mask is not None:
            logits += attn_mask
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorn1 = nn.LayerNorm(config.dim)
        self.attn = Attention(
            config.dim,
            heads = config.heads,
            dim_key = config.attn_dim_key,
            dim_value = config.dim // config.heads,
            dropout = config.attn_dropout,
            pos_dropout = config.pos_dropout,
            num_rel_pos_features = config.dim // config.heads
        )
        self.attn_dropout = nn.Dropout(config.dropout_rate)

        self.ffn = Residual(nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.Dropout(config.dropout_rate)
        ))

    def forward(self, x, attn_mask=None):
        x1 = self.LayerNorn1(x)
        x2 = self.attn(x1, attn_mask)
        x3 = self.attn_dropout(x2)
        x4 = x + x3

        x5 = self.ffn(x4)

        return x5


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(0.01 * hidden_states + input_tensor)
        return hidden_states   
    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.act_fn = nn.ReLU()
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states += input_tensor
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class CrossAttnLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.crossattention = BertSelfAttention(
            config, is_cross_attention=True
        )

        # self.intermediate_query = BertIntermediate(config)
        self.output_query = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None, # None
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False, # False
    ):
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]
        # intermediate_output = self.intermediate_query(query_attention_output)
        # layer_output = self.output_query(intermediate_output, query_attention_output)
        layer_output = self.output_query(query_attention_output, hidden_states)

        return layer_output



# main class

class Enformer(PreTrainedModel):
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        # self.use_label_attention=config.use_label_attention
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.dim // config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = config.dim // config.heads
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        # create trunk sequential module

        self._trunk = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # create final heads for human and mouse



        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

        # if self.use_label_attention:
        #     self.dim_value_label = 36
        #     self.dim_key_label = 9
        #     self.multi_label_layer = nn.Sequential(
        #         nn.LayerNorm(config.dim),
        #         Multi_Label_Attention(
        #             self.dim,
        #             heads = config.num_label,
        #             dim_key = 8,
        #             dim_value = 36,
        #             dropout = config.attn_dropout,
        #             pos_dropout = config.pos_dropout,
        #             num_rel_pos_features = 36
        #         ),
        #         nn.Dropout(config.dropout_rate),
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             nn.Linear(36, 36 * 2),
        #             nn.Dropout(config.dropout_rate),
        #             nn.ReLU(),
        #             nn.Linear(36 * 2, 36),
        #             nn.Dropout(config.dropout_rate)
        #         ))
        #     )
        #     self.cross_label_layer = nn.Sequential(
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             Label_Cross_Attention(
        #                 36,
        #                 dim_key = 8,
        #                 dim_value = 36,
        #                 dropout = config.attn_dropout,
        #                 pos_dropout = config.pos_dropout,
        #                 num_rel_pos_features = 36
        #             ),
        #             nn.Dropout(config.dropout_rate)
        #         )),
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             nn.Linear(36, 36 * 2),
        #             nn.Dropout(config.dropout_rate),
        #             nn.ReLU(),
        #             nn.Linear(36 * 2, 36),
        #             nn.Dropout(config.dropout_rate)
        #         ))
        #     )
        #     self.out_layer = nn.Linear(36, 1)

    def add_heads(self, **kwargs):
        self.output_heads = kwargs

        # if self.use_label_attention:
        #     self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
        #     nn.Linear(self.dim * 2, features),
        #     GELU(),
        #     LabelAttention(features)
        #     # nn.Softplus()
        #     # nn.Sigmoid()
        # ), kwargs))

        # else:
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            
            # nn.Softplus()
            # nn.Sigmoid()
        ), kwargs))

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.stem(x)
        x = self.conv_tower(x)
        x = rearrange(x, 'b d n -> b n d')
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        target = None,
        return_corr_coef = False,
        return_embeddings = False,
        return_only_embeddings = False,
        head = None,
        target_length = None,
        return_only_embeddings_before_final_pointwise = False,
        return_both = False,
        final_res=False
    ):
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        if exists(target_length):
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk

        x = trunk_fn[0](x)
        x = trunk_fn[1](x)
        x = trunk_fn[2](x)
        x = trunk_fn[3](x)
        x = trunk_fn[4](x)
        if return_only_embeddings_before_final_pointwise:
            return x
        x0 = x
        x = trunk_fn[5](x)
        x = trunk_fn[6](x)
        # x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, '() ... -> ...')

        if return_only_embeddings:
            return x
        
        
        # if self.use_label_attention:
        #     out = self.multi_label_layer(x)
        #     # out = self.cross_label_layer(out)
        #     out = self.out_layer(out)
        #     out = rearrange(out, "b n h d -> b n (h d)")
        #     out = {
        #         'rbp': out
        #     }
        # else:
        out = map_values(lambda fn: fn(x), self._heads)

        # if exists(head):
        #     assert head in self._heads, f'head {head} not found'
        #     out = out[head]

        # if exists(target):
        #     assert exists(head), 'head must be passed in if one were to calculate loss directly with targets'

        #     if return_corr_coef:
        #         return pearson_corr_coef(out, target)

        #     return poisson_loss(out, target)

        # if return_embeddings:
        #     return out, x
        if return_both:
            return out['rbp'], x0

        return out
    

class CrossEnformer(Enformer):
    def __init__(self, config):
        super().__init__(config)
        bert_name = '/data0/liyichong/binding_data/scibert_scivocab_uncased/'
        encoder_config = BertConfig.from_pretrained(bert_name)
        self.cell_line_emb_layer = nn.Embedding(num_embeddings=config.num_label, embedding_dim=config.cell_line_dim)
        # self.use_label_attention=config.use_label_attention
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2
        self.cross_freq = config.cross_freq
        self.encoder_config = encoder_config

        # 修改bert参数
        encoder_config.hidden_size = config.dim
        encoder_config.encoder_width = config.esm_dim + config.cell_line_dim

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding = 7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size = 2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size = 5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size = 2)
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.dim // config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = config.dim // config.heads
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            )
            # TransformerLayer(config)
            )

        self.transformer = nn.Sequential(*transformer)

        self.cross_attn = []
        self.cross_attn_index = []
        cross_attn_cnt = 0
        for i in range(config.depth):
            if i % self.cross_freq == 0:
                self.cross_attn.append(
                    CrossAttnLayer(
                        config=encoder_config
                    )
                )
                self.cross_attn_index.append(cross_attn_cnt)
                cross_attn_cnt += 1
            else:
                self.cross_attn_index.append(None)
        self.cross_attn = nn.Sequential(*self.cross_attn)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        self._trunk = nn.Sequential(
            Rearrange('b n d -> b d n'),
            self.stem,
            self.conv_tower,
            Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        # self._trunk_1 = nn.Sequential(
        #     Rearrange('b n d -> b d n'),
        #     self.stem,
        #     self.conv_tower,
        #     Rearrange('b d n -> b n d'),
        # )

        # self._trunk_2 = nn.Sequential(
        #     self.crop_final,
        #     self.final_pointwise
        # )
        # create final heads for human and mouse

        # self.line_protein_k = nn.Linear((config.esm_dim+config.cell_line_dim), config.dim*2)
        # self.line_protein_v = nn.Linear((config.esm_dim+config.cell_line_dim), config.dim*2)

        if not config.use_bi_cross:
            self.outlayer = CrossAttentionOutLayer(config)
        else:
            self.outlayer = Bi_CrossAttentionOutLayer(config)

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

        # if self.use_label_attention:
        #     self.dim_value_label = 36
        #     self.dim_key_label = 9
        #     self.multi_label_layer = nn.Sequential(
        #         nn.LayerNorm(config.dim),
        #         Multi_Label_Attention(
        #             self.dim,
        #             heads = config.num_label,
        #             dim_key = 8,
        #             dim_value = 36,
        #             dropout = config.attn_dropout,
        #             pos_dropout = config.pos_dropout,
        #             num_rel_pos_features = 36
        #         ),
        #         nn.Dropout(config.dropout_rate),
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             nn.Linear(36, 36 * 2),
        #             nn.Dropout(config.dropout_rate),
        #             nn.ReLU(),
        #             nn.Linear(36 * 2, 36),
        #             nn.Dropout(config.dropout_rate)
        #         ))
        #     )
        #     self.cross_label_layer = nn.Sequential(
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             Label_Cross_Attention(
        #                 36,
        #                 dim_key = 8,
        #                 dim_value = 36,
        #                 dropout = config.attn_dropout,
        #                 pos_dropout = config.pos_dropout,
        #                 num_rel_pos_features = 36
        #             ),
        #             nn.Dropout(config.dropout_rate)
        #         )),
        #         Residual(nn.Sequential(
        #             nn.LayerNorm(36),
        #             nn.Linear(36, 36 * 2),
        #             nn.Dropout(config.dropout_rate),
        #             nn.ReLU(),
        #             nn.Linear(36 * 2, 36),
        #             nn.Dropout(config.dropout_rate)
        #         ))
        #     )
        #     self.out_layer = nn.Linear(36, 1)

    def _strip_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉'module.'
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_enformer(self, model_dir, freeze=True):
        state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.'前缀
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        if freeze:
            for param in self.enformer.parameters():
                param.requires_grad = False
            print("freeze RBP Former backbone.")
        print("Load RBP Former backbone parameters from {}.".format(model_dir))

    def load_pretrained(self, 
                checkpoint_path: str, 
                freeze_loaded: bool = True,  # 新增参数
                disable_dropout: bool = True,
                strict: bool = False,
                verbose: bool = True) -> Set[str]:
        """
        加载预训练权重并可选择性地冻结这些参数
        
        Args:
            checkpoint_path: 权重文件路径
            freeze_loaded: 是否冻结加载的参数
            strict: 是否严格匹配权重
            verbose: 是否打印详细信息
        """
        try:
            # 加载权重文件
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的权重文件格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 处理前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 获取当前模型的参数名
            model_state = self.state_dict()
            model_keys = set(model_state.keys())
            checkpoint_keys = set(state_dict.keys())
            
            # 找出匹配的键
            matched_keys = model_keys & checkpoint_keys
            missing_keys = model_keys - checkpoint_keys
            
            # 加载参数
            if not strict:
                new_state_dict = {k: state_dict[k] for k in matched_keys}
                self.load_state_dict(new_state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=True)
            
            # 冻结加载的参数
            frozen_modules = set()
            if freeze_loaded:
                frozen_params = 0
                # 从参数名推断模块路径
                for name in matched_keys:
                    # 将参数名转换为模块路径 (例如: layer1.0.conv.weight -> layer1.0)
                    module_path = '.'.join(name.split('.')[:-1])
                    frozen_modules.add(module_path)

            # 处理dropout
            if disable_dropout and frozen_modules:
                dropout_count = 0
                # 递归处理所有模块
                def _handle_dropout(module, prefix=''):
                    nonlocal dropout_count
                    for name, child in module.named_children():
                        full_path = f"{prefix}.{name}" if prefix else name
                        # 如果当前模块路径在被冻结的路径中
                        if any(full_path.startswith(frozen_path) for frozen_path in frozen_modules):
                            # 禁用所有dropout层
                            if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                                child.p = 0
                                dropout_count += 1
                            # 递归处理子模块
                            _handle_dropout(child, full_path)
                
                _handle_dropout(self)
            
            # 打印信息
            if verbose:
                print(f"Successfully loaded {len(matched_keys)} parameters")
                if freeze_loaded:
                    print(f"Froze {frozen_params} parameters")
                if missing_keys:
                    print(f"Warning: Missing {len(missing_keys)} parameters:")
                    for key in sorted(missing_keys):
                        print(f"  {key}")
            self.pretrained_matched_keys = matched_keys
            
            return matched_keys
                
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise

    def get_param_groups(self):
        # 获取预训练加载的参数名集合
        pretrained_param_names = getattr(self, 'pretrained_matched_keys', set())
        
        # 初始化两个参数组
        pretrained_params = []
        new_params = []
        
        # 遍历所有参数
        for name, param in self.named_parameters():
            if param.requires_grad:  # 只处理需要梯度的参数
                if name in pretrained_param_names:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)
        
        return {
            'pretrained': pretrained_params,
            'new': new_params
        }

    # 添加一个用于查看参数冻结状态的方法
    def print_parameter_status(self):
        """打印所有参数的冻结状态"""
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

    # 添加解冻方法
    def unfreeze_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


    def forward(self, x,
                x_mask, 
                esm2_reps, 
                cell_line_tensor,
                pro_mask=None, 
                target_length = None,
                head_mask=None,
                return_only_embeddings=False
            ):
        # 蛋白质信息编码
        cell_line_emb = self.cell_line_emb_layer(cell_line_tensor)
        protein_reps = torch.cat([esm2_reps, cell_line_emb], dim=2)

        # Enformer 降采样阶段
        if isinstance(x, list):
            x = str_to_one_hot(x)

        elif x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)

        no_batch = x.ndim == 2

        if no_batch:
            x = rearrange(x, '... -> () ...')

        if exists(target_length):
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn[0](x)
        x = trunk_fn[1](x)
        x = trunk_fn[2](x)
        x = trunk_fn[3](x)

        # cross transformer 阶段
        
        # x 的musk
        input_shape = x.size()[:-1]
        device = x.device
        batch_size, seq_length = input_shape
        
        if x_mask is None:
            x_mask = torch.ones(
                ((batch_size, seq_length)), device=device
            )
        x_extended_attention_mask = self.get_extended_attention_mask(
            x_mask, input_shape, device, is_decoder=False
        )

        # protein 的mask，其实没有
        (
            encoder_batch_size,
            encoder_sequence_length,
            _,
        ) = protein_reps.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if pro_mask is None:
            pro_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            pro_extended_attention_mask = self.invert_attention_mask(
                pro_attention_mask
            )
        
        head_mask = self.get_head_mask(head_mask, self.encoder_config.num_hidden_layers)

        for self_attn, index in zip(self.transformer, self.cross_attn_index):
            x = self_attn(x) #@, x_extended_attention_mask)
            if index is not None:
                x = self.cross_attn[index](
                    hidden_states=x,
                    attention_mask=x_extended_attention_mask,
                    encoder_hidden_states=protein_reps,
                    encoder_attention_mask=pro_extended_attention_mask
                )

        x = trunk_fn[5](x)
        x = trunk_fn[6](x)

        out = self.outlayer(x, protein_reps)
        return out

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        device: device,
        is_decoder: bool,
        has_query: bool = False,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :].to(device)
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.to(device)
        return extended_attention_mask


class Bi_CrossAttentionOutLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim_model1 = config.dim * 2
        dim_model2 = config.esm_dim+config.cell_line_dim
        num_heads = config.heads
        dropout = config.attn_dropout
        assert dim_model1 % num_heads == 0
        assert dim_model2 % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim_model1 // num_heads
        
        # x1 -> x2 方向的投影矩阵
        self.q1_proj = nn.Linear(dim_model1, dim_model1)  # [batch_size, l1, dim_model1] -> [batch_size, l1, dim_model1]
        self.k2_proj = nn.Linear(dim_model2, dim_model1)  # [batch_size, l2, dim_model2] -> [batch_size, l2, dim_model1]
        
        # x2 -> x1 方向的投影矩阵
        self.q2_proj = nn.Linear(dim_model2, dim_model2)  # [batch_size, l2, dim_model2] -> [batch_size, l2, dim_model2]
        self.k1_proj = nn.Linear(dim_model1, dim_model2)  # [batch_size, l1, dim_model1] -> [batch_size, l1, dim_model2]
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
        # 两个方向注意力分数的组合权重
        self.attention_weights = nn.Parameter(torch.ones(2))
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x1, x2):
        """
        输入:
        x1: [batch_size, l1, dim_model1]
        x2: [batch_size, l2, dim_model2]
        输出:
        logits: [batch_size, l1, l2]  # 每对位置之间的匹配分数
        """
        batch_size = x1.shape[0]
        l1, l2 = x1.shape[1], x2.shape[1]
        
        # 1. x1 -> x2 方向
        # [batch_size, num_heads, l1, head_dim]
        Q1 = self.q1_proj(x1).view(batch_size, l1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, num_heads, l2, head_dim]
        K2 = self.k2_proj(x2).view(batch_size, l2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # attention_scores1: [batch_size, num_heads, l1, l2]
        attention_scores1 = torch.matmul(Q1, K2.permute(0, 1, 3, 2)) / self.scale
        attention_scores1 = self.dropout(attention_scores1)
        
        # 2. x2 -> x1 方向
        # [batch_size, num_heads, l2, head_dim]
        Q2 = self.q2_proj(x2).view(batch_size, l2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, num_heads, l1, head_dim]
        K1 = self.k1_proj(x1).view(batch_size, l1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # attention_scores2: [batch_size, num_heads, l2, l1]
        attention_scores2 = torch.matmul(Q2, K1.permute(0, 1, 3, 2)) / self.scale
        attention_scores2 = self.dropout(attention_scores2)
        
        # 3. 对多个头的分数取平均
        # [batch_size, l1, l2]
        scores1 = attention_scores1.mean(dim=1)
        # [batch_size, l2, l1] -> [batch_size, l1, l2]
        scores2 = attention_scores2.mean(dim=1).transpose(-1, -2)
        
        # 4. 组合两个方向的分数
        weights = self.softmax(self.attention_weights)
        logits = weights[0] * scores1 + weights[1] * scores2
        
        return logits
    


class CrossAttentionOutLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cell_line_emb_layer = nn.Embedding(num_embeddings=config.num_label, embedding_dim=config.cell_line_dim)
        # self.corss_attn = Label_Cross_Attention(encoder_config)
        self.line_protein_k = nn.Linear((config.esm_dim+config.cell_line_dim), config.attn_dim_key * config.heads)
        self.rna_q = nn.Linear(config.dim * 2, config.attn_dim_key * config.heads)


        self.dropout = nn.Dropout(config.attn_dropout)
        self.scale = config.attn_dim_key ** -0.5
        self.rel_content_bias = nn.Parameter(torch.randn(1, config.heads, 1, config.attn_dim_key))
        self.heads = config.heads
    def forward(self, rna_reps, protein_reps):
        n, h, device = rna_reps.shape[-2], self.heads, rna_reps.device
        k = self.line_protein_k(protein_reps)
        q = self.rna_q(rna_reps)

        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k))
        q = q * self.scale
        # content_logits = einsum('b h i d, b h j d -> b h i j', q, k)
        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)
        content_logits = torch.mean(content_logits, dim=1)
        content_logits = self.dropout(content_logits)

        return content_logits


class Enformer2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enformer = Enformer(config)

        if not config.use_bi_cross:
            self.outlayer = CrossAttentionOutLayer(config)
        else:
            self.outlayer = Bi_CrossAttentionOutLayer(config)

    def load_enformer(self, model_dir, freeze=True):
        state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.'前缀
            else:
                new_state_dict[k] = v

        self.enformer.load_state_dict(new_state_dict)
        if freeze:
            for param in self.enformer.parameters():
                param.requires_grad = False
            print("freeze RBP Former backbone.")
        print("Load RBP Former backbone parameters from {}.".format(model_dir))


    def forward(self, x, esm2_reps, cell_line_tensor):
        cell_line_emb = self.cell_line_emb_layer(cell_line_tensor)
        protein_reps = torch.cat([esm2_reps, cell_line_emb], dim=2)
        # protein_reps = protein_reps.unsqueeze(0).expand(x.shape[0], -1, -1)
        rna_reps = self.enformer(x, return_only_embeddings=True)
        content_logits = self.outlayer(rna_reps, protein_reps)
        return content_logits


class CrossAttentionWithFFN(nn.Module):
    def __init__(self, config, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.d_model = config.d_model  # Embedding dimension
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        # Cross-Attention components
        self.query_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=config.num_attention_heads)
        self.key_value_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=config.num_attention_heads)

        # FFN components
        self.ffn1 = nn.Linear(self.d_model, self.d_model * 4)  # First layer of FFN
        self.ffn2 = nn.Linear(self.d_model * 4, self.d_model)  # Second layer of FFN
        self.activation = nn.ReLU()

        # Optional layer norm (often used in Transformer-like architectures)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """
        hidden_states: (B, query_len, D) - The query vectors (e.g., decoder hidden states)
        encoder_hidden_states: (B, key_len, D) - The key and value vectors (e.g., encoder hidden states)
        attention_mask: (B, query_len, key_len) - Optional attention mask
        """
        # Cross-Attention computation
        query_len, key_len = hidden_states.size(1), encoder_hidden_states.size(1)
        
        # Cross-attention (query with key-value)
        attention_output, attention_weights = self.query_attention(
            hidden_states.transpose(0, 1),  # Shape: (query_len, B, D)
            encoder_hidden_states.transpose(0, 1),  # Shape: (key_len, B, D)
            encoder_hidden_states.transpose(0, 1),  # Value is same as key in typical self-attention
            key_padding_mask=attention_mask,  # Optional: apply attention mask (e.g., padding)
        )

        # attention_output shape: (query_len, B, D)
        attention_output = attention_output.transpose(0, 1)  # Back to (B, query_len, D)

        # Apply FFN (Feed Forward Network)
        # Layer normalization + dropout can be applied before or after FFN, depending on your architecture
        attention_output = self.layer_norm1(attention_output)
        ffn_output = self.ffn1(attention_output)  # Shape: (B, query_len, D * 4)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn2(ffn_output)  # Shape: (B, query_len, D)

        # Dropout + Layer normalization after FFN
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.layer_norm2(ffn_output)

        return ffn_output, attention_weights
    

def cross_attention(Q, K, V, attention_mask=None):
    """
    Q: (B, 1024, D)
    K: (B, 166, D)
    V: (B, 166, D)
    attention_mask: (B, 1024, 166) (optional, used for masking)
    """

    # Step 1: Calculate the attention scores
    d_k = Q.size(-1)  # The dimension of the query/key vectors (D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5  # (B, 1024, 166)

    # Step 2: Apply attention mask (optional)
    if attention_mask is not None:
        scores = scores + attention_mask  # Add mask (usually -inf for invalid positions)

    # Step 3: Apply softmax to get attention weights
    # attention_weights = F.softmax(scores, dim=-1)  # (B, 1024, 166)
    attention_weights = scores

    # Step 4: Calculate the weighted sum of values
    # output = torch.matmul(attention_weights, V)  # (B, 1024, D)

    return attention_weights


class LabelAttention(nn.Module):
    def __init__(self, num_labels):
        super(LabelAttention, self).__init__()
        self.attention = nn.Linear(num_labels, num_labels)  # 在特征维度（N）上生成注意力权重
        self.softmax = nn.Softmax(dim=-1)  # 对最后一个维度应用 Softmax

    def forward(self, x):
        # x shape: (B, L, N)
        attention_scores = self.attention(x)  # 计算注意力权重 (B, L, N)
        attention_weights = self.softmax(attention_scores)  # 应用 Softmax 得到注意力分布 (B, L, N)
        
        # 使用注意力权重对原始输出进行加权
        weighted_output = x * attention_weights  # 对 N 维度加权 (B, L, N)

        return weighted_output
    

class Multi_Label_Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)
        self.to_res = nn.Linear(dim, heads * dim_value, bias = False)

        # self.to_out = nn.Linear(dim_value * heads, dim)
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)#.half()
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n h d')

        x_res = rearrange(self.to_res(x), 'b n (h d) -> b n h d', h=h)
        out += x_res
        return out #self.to_out(out)
    

class Label_Cross_Attention(nn.Module):
    def __init__(self, dim, num_rel_pos_features, dim_key=64, dim_value=64, dropout=0., pos_dropout=0.01):
        super(Label_Cross_Attention, self).__init__()
        self.scale = dim_key ** -0.5

        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim_key, bias=False)
        self.to_k = nn.Linear(dim, dim_key, bias=False)
        self.to_v = nn.Linear(dim, dim_value, bias=False)

        # Output projection
        self.to_out = nn.Linear(dim_value, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # Relative positional encoding
        self.num_rel_pos_features = num_rel_pos_features
        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, 1, dim_key))

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(pos_dropout)

    def forward(self, x):
        b, n, num_labels, d = x.shape

        # Project inputs to Q, K, V
        q = self.to_q(x)  # (B, n, num_labels, dim_key)
        k = self.to_k(x)  # (B, n, num_labels, dim_key)
        v = self.to_v(x)  # (B, n, num_labels, dim_value)

        q = q * self.scale

        # Content logits (Q*K^T)
        content_logits = einsum('b n i d, b n j d -> b n i j', q + self.rel_content_bias, k)

        # Relative positional encoding
        positions = get_positional_embed(num_labels, self.num_rel_pos_features, x.device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)  # (num_labels, dim_key)
        rel_logits = einsum('b n i d, j d -> b n i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        # Combine logits and compute attention weights
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Compute output
        out = einsum('b n i j, b n j d -> b n i d', attn, v)  # (B, n, num_labels, dim_value)
        return self.to_out(out)  # (B, n, num_labels, dim)