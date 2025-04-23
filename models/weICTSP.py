import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
import random

import pywt


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, N=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta, N=N
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, N=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, N=N)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, N=N)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, N=N)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            # self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [x.shape[1], 1, 1]), requires_grad=requires_grad)
            # self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [x.shape[1], 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size

            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            h0 = h0.to('cuda')
            h1 = h1.to('cuda')
            # print(f"h0 device: {h0.device}, input device: {approx_coeffs_pad.device}")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        for i in range(m):
            detail_coeff = detail_coeffs[:, :, i, :]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            g0 = g0.to('cuda')
            g1 = g1.to('cuda')
            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2

        return approx_coeff


class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5):
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        self.requires_grad = requires_grad
        self.wv = wv
        self.m = m
        self.kernel_size = kernel_size

        # self.swt = None#WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.swt = nn.Identity()  
        self.out_wavelet = nn.Identity()

        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            None,
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, N=None):
        #  WaveletEmbedding
        new_swt = WaveletEmbedding(
            d_channel=N,
            swt=True,
            requires_grad=self.requires_grad,
            wv=self.wv,
            m=self.m,
            kernel_size=self.kernel_size
        )
        self.swt = new_swt
        self.add_module("swt", new_swt)  

        new_out_wavelet = WaveletEmbedding(
            d_channel=N,
            swt=False,
            requires_grad=self.requires_grad,
            wv=self.wv,
            m=self.m,
            kernel_size=self.kernel_size
        )
        self.out_projection[1] = new_out_wavelet
        self.add_module("out_wavelet", new_out_wavelet)

        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = self.query_projection(queries).permute(0, 3, 2, 1)
        keys = self.key_projection(keys).permute(0, 3, 2, 1)
        values = self.value_projection(values).permute(0, 3, 2, 1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0, 3, 2, 1))

        return out, attn


class GeomAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False,
                 alpha=1., ):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.alpha = alpha

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries ** 2, dim=-1)
        keys_norm2 = torch.sum(keys ** 2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)  # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2  # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return (V.contiguous(), scores.abs().mean())


class Tokenizer(nn.Module):
    def __init__(self, lookback=96, output=96, stride=None):
        super(Tokenizer, self).__init__()
        self.d = lookback + output  # length of chunks
        self.s = output if stride is None else stride  # stride

    def forward(self, tensor):
        # tensor: B C L
        # return: B C N d
        return tensor.flip(-1).unfold(dimension=2, size=self.d, step=self.s).flip(-1).flip(-2)


def normalize_vectors(x):
    norms = torch.norm(x, p=2, dim=-1, keepdim=True)
    normalized_x = x / norms
    return normalized_x


class TokenFusionLayerWithAdjustableRatio(nn.Module):
    def __init__(self, input_dim, fusion_ratio=0.25, init_reduce_rate=4, batch_size=1024):
        super(TokenFusionLayerWithAdjustableRatio, self).__init__()
        self.input_dim = input_dim
        self.fusion_ratio = fusion_ratio
        self.init_reduce_rate = init_reduce_rate
        self.batch_size = batch_size
        self.input_q = nn.Linear(input_dim // 2, 32)
        self.input_k = nn.Linear(input_dim // 2, 32)
        self.silu = nn.ReLU()

    def forward(self, x, num_target_tokens, limit=512):
        target_tokens = x[:, -num_target_tokens:]
        other_tokens = x[:, :-num_target_tokens]

        attention_scores = torch.zeros((x.size(0), other_tokens.size(1)), device=x.device)

        for start_idx in range(0, other_tokens.size(1), self.batch_size):
            end_idx = min(start_idx + self.batch_size, other_tokens.size(1))
            batch_other_tokens = other_tokens[:, start_idx:end_idx]
            batch_other_tokens_expanded = normalize_vectors(
                self.input_q(batch_other_tokens[:, :, -batch_other_tokens.shape[-1] // 2:])).unsqueeze(1)

            cosine_sims = F.cosine_similarity(batch_other_tokens_expanded, normalize_vectors(
                self.input_k(target_tokens[:, :, -batch_other_tokens.shape[-1] // 2:])).unsqueeze(2), dim=-1)
            cosine_sims = self.silu(cosine_sims)

            attention_scores[:, start_idx:end_idx] = cosine_sims.mean(dim=1)

        sorted_scores, sorted_indices = torch.sort(attention_scores, descending=True, dim=1)

        num_keep = int(other_tokens.size(1) * self.fusion_ratio)
        top_indices = sorted_indices[:, :num_keep]
        top_tokens = other_tokens.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, other_tokens.size(2)))

        top_scores = attention_scores.gather(1, top_indices)
        fused_tokens = [top_tokens * top_scores.unsqueeze(-1)]

        remaining_indices = sorted_indices[:, num_keep:]
        remaining = limit - num_keep
        fusion_step = self.init_reduce_rate

        while remaining > 0 and fusion_step <= remaining:
            num_to_fuse = max(int(remaining * self.fusion_ratio), fusion_step)
            num_groups = num_to_fuse // fusion_step

            group_tokens = torch.zeros((x.size(0), num_groups, x.size(2)), device=x.device)
            group_scores = torch.zeros((x.size(0), num_groups), device=x.device)

            for i in range(num_groups):
                start_idx = i * fusion_step
                end_idx = start_idx + fusion_step
                slice_tokens = other_tokens.gather(1,
                                                   remaining_indices[:, start_idx:end_idx].unsqueeze(-1).expand(-1, -1,
                                                                                                                other_tokens.size(
                                                                                                                    2)))
                slice_scores = sorted_scores.gather(1, remaining_indices[:, start_idx:end_idx])
                weights = F.softmax(slice_scores, dim=1)
                group_tokens[:, i, :] = torch.sum(slice_tokens * weights.unsqueeze(-1), dim=1)
                group_scores[:, i] = torch.sum(slice_scores * weights, dim=1)

            fused_tokens.append(group_tokens * group_scores.unsqueeze(-1))
            remaining_indices = remaining_indices[:, num_to_fuse:]
            remaining -= num_to_fuse
            fusion_step *= self.init_reduce_rate

        output_tokens = torch.cat(fused_tokens, dim=1)[:, 0:limit, :]
        output_tokens = torch.cat([torch.flip(output_tokens, [1]), target_tokens], dim=1)
        return output_tokens


class ICTSP(nn.Module):
    def __init__(self, lookback=96, output=96, depth=3, heads=8, mlp_ratio=4, d_model=0, emb_init=0.01,
                 output_projection=False, external_stride=16, external_context=False, task_emb_dim=0, n_channels=0,
                 channel_emb_dim=0, partial_mask=True, dropout=0.1, inter_series_latent_dim=16, number_of_targets=0,
                 time_emb_dim=0, token_retriever_flag=True, linear_warmup_steps=5000, token_limit=1024,
                 ICL_embedding=False,
                 mask_length_for_comparison=0, alpha=0.5, m=5, wv='db4', factor=1, kernel_size=3, geomattn_dropout=0.1,
                 d_ff=256,
                 activation='gelu', output_attention=False, requires_grad=False):
        super(ICTSP, self).__init__()
        self.lookback = lookback
        self.pred_len = output
        self.time_emb_dim = time_emb_dim

        self.lookback_pool = [lookback]
        self.future_pool = [output]
        self.external_stride = external_stride
        self.ICL_embedding = ICL_embedding

        self.factor = factor
        self.alpha = alpha
        self.m = m
        self.wv = wv
        # self.dec_in = dec_in
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.geomattn_dropout = geomattn_dropout
        self.output_attention = output_attention
        self.requires_grad = requires_grad
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = depth
        if ICL_embedding:
            self.x_projection = nn.ModuleDict({str(lb): nn.Linear(lb, d_model // 2) for lb in self.lookback_pool})
            self.y_projection = nn.ModuleDict({str(ft): nn.Linear(ft, d_model // 2) for ft in self.future_pool})
        else:
            self.input_projection = nn.ModuleDict(
                {str(lb) + '_' + str(ft): nn.Linear(lb + ft, d_model) for lb in self.lookback_pool for ft in
                 self.future_pool})
        emb_size = d_model
        encoder = Encoder(
            [
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, self.factor, attention_dropout=self.geomattn_dropout,
                            output_attention=self.output_attention, alpha=self.alpha
                        ),
                        self.d_model,
                        requires_grad=self.requires_grad,
                        wv=self.wv,
                        m=self.m,
                        # d_channel=self.dec_in,
                        kernel_size=self.kernel_size,
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.geomattn_dropout,
                    activation=self.activation,
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.transformer_encoder = encoder
        # self.transformer_encoder = TransformerEncoder(emb_size, depth, heads, mlp_ratio, dropout=dropout)

        self.input_norm = nn.LayerNorm(emb_size)
        self.output_norm = nn.LayerNorm(emb_size)

        self.output_embedding = nn.Parameter(emb_init * torch.randn(1, 1, 1200))
        self.output_projection = nn.ModuleDict({str(ft): nn.Linear(emb_size, ft) for ft in self.future_pool})

        self.partial_mask = partial_mask

        self.n_channels = n_channels + time_emb_dim
        self.n_heads = heads

        self.channel_discerning_mask = nn.Parameter(emb_init * torch.randn(1, 1024, emb_size))

        self.number_of_targets = number_of_targets

        self.in_context_learning_type = 'concat'  # ['concat', 'seperate']
        self.in_context_positional_embedding = nn.Parameter(emb_init * torch.randn(1, 8192, 1, emb_size))
        self.in_context_positional_embedding_after = nn.Parameter(emb_init * torch.randn(1, 8192, emb_size))

        self.external_stride = external_stride

        self.number_of_targets = number_of_targets

        self.initialized = False

        self.token_retriever_flag = token_retriever_flag
        self.linear_warmup_steps = linear_warmup_steps
        self.token_merger = TokenFusionLayerWithAdjustableRatio(emb_size, 0.1, 8, batch_size=2048)

        self.token_limit = token_limit

        self.linear_warm_up_counter = 0

        self.mask_length_for_comparison = mask_length_for_comparison
        # self.linear_refill = nn.Parameter(torch.zeros(1, mask_length_for_comparison, 128)) # nn.Linear(512, mask_length_for_comparison)
        self.linear_refill = nn.Linear(512, mask_length_for_comparison)
        self.linear_refill_norm = nn.LayerNorm(mask_length_for_comparison)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=1e-5/np.sqrt(2 * depth)) # 0.02/math.sqrt(2 * config.n_layer)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False):
        # x: B L_I C
        if self.mask_length_for_comparison != 0:
            # B, L, C = x.shape
            # mean, std = x[:, [self.mask_length_for_comparison]], x[:, self.mask_length_for_comparison:].std(dim=1, keepdim=True) + 1e-8#x[:, self.mask_length_for_comparison:].mean(dim=1, keepdim=True), x[:, self.mask_length_for_comparison:].std(dim=1, keepdim=True) + 1e-8
            # refill_input = ((x[:, self.mask_length_for_comparison:] - mean)/std).permute(0, 2, 1)
            # linear_refill = self.linear_refill_norm(self.linear_refill(refill_input)).permute(0, 2, 1)
            # linear_refill = linear_refill * std + mean
            # x[:, 0:self.mask_length_for_comparison] = linear_refill
            B, L, C = x.shape
            x[:, 0:self.mask_length_for_comparison] = x[:, self.mask_length_for_comparison:].mean(dim=1,
                                                                                                  keepdim=True).expand(
                -1, self.mask_length_for_comparison,
                -1)  # x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#x[:, [self.mask_length_for_comparison]].expand(-1, self.mask_length_for_comparison, -1)#x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#x[:, self.mask_length_for_comparison:].mean(dim=1, keepdim=True).expand(-1, self.mask_length_for_comparison, -1)#x[:, [-1]].expand(-1, self.mask_length_for_comparison, -1)#self.linear_refill(x[:, self.mask_length_for_comparison:].permute(0, 2, 1)).permute(0, 2, 1)

        lookback = self.lookback
        future = x_mark_dec.shape[1] - x.shape[1] if x_mark_dec is not None else self.pred_len
        mean = x[:, [-1], :].detach()
        std = 1

        x = x.permute(0, 2, 1)  # B C L_I
        output_embedding = self.output_embedding[:, :, 0:future].expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, output_embedding + mean.permute(0, 2, 1)], dim=-1)  # B C L

        number_of_targets = x.shape[1] if (
                    self.number_of_targets == 0 or x.shape[1] != self.n_channels) else self.number_of_targets

        x_orig = x[:, :, 0:-future].clone()
        if self.training:
            shifting = random.randint(0, self.external_stride)
            if shifting != 0:
                x_orig = x_orig[:, :, 0:-shifting]

        B, C, _ = x.shape

        x_target = x[:, -number_of_targets:, -(lookback + future):]

        external_tokenizer = Tokenizer(lookback, future, stride=self.external_stride)
        ex_tokens = external_tokenizer(x_orig)  # if self.training else self.tokenizer(x_orig)   # B C_ex N_ex d
        _, _, _, d = ex_tokens.shape

        ex_tokens = ex_tokens.permute(0, 2, 1, 3).reshape(B, -1, d)  # B N_ex*C_ex d

        x_tokens = torch.cat([ex_tokens, x_target], dim=1)  # B N_ex*C_ex+N*C d
        token_mean = x_tokens[:, :, [-(future + 1)]].detach()

        token_std = x_tokens[:, :, 0:-future].std(dim=-1, keepdim=True) + 1e-7

        x_tokens = x_tokens - token_mean

        x_tokens_orig = x_tokens.clone()

        if self.in_context_learning_type == 'concat':
            if self.ICL_embedding:
                x_tokens = torch.cat(
                    [self.x_projection[str(lookback)](x_tokens[:, :, 0:-future]),  # why str? if tokenized by wavelet?
                     self.y_projection[str(future)](x_tokens[:, :, -future:])], dim=-1)  # B N*C d
            else:
                x_tokens = self.input_projection[str(lookback) + '_' + str(future)](x_tokens)  # B N*C d

        channel_discerning_mask = self.channel_discerning_mask[:, -C:, :]
        x_tokens = x_tokens + channel_discerning_mask.repeat(1, x_tokens.shape[1] // C, 1)

        in_context_positional_embedding = self.in_context_positional_embedding[:, -(x_tokens.shape[1] // C):, :,
                                          :].expand(-1, -1, C, -1)
        in_context_positional_embedding = in_context_positional_embedding.reshape(
            in_context_positional_embedding.shape[0],
            in_context_positional_embedding.shape[1] * in_context_positional_embedding.shape[2],
            in_context_positional_embedding.shape[3])
        x_tokens = x_tokens + in_context_positional_embedding

        if self.linear_warm_up_counter < self.linear_warmup_steps:
            if self.training:
                self.linear_warm_up_counter += 1
            x_output = self.output_projection[str(future)](x_tokens[:, -number_of_targets:, :])
            x_tokens = self.input_norm(x_tokens)

            x_tokens = self.output_norm(x_tokens)
            x_output = x_output + token_mean[:, -x_output.shape[1]:, :]
            x_output = x_output.permute(0, 2, 1)
            return x_output

        if not self.initialized:
            print('Token Size: ', x_tokens.shape)

        x_tokens_orig = x_tokens.clone()

        if self.token_retriever_flag:
            x_tokens = self.token_merger(x_tokens, number_of_targets)

        x_tokens_merged = x_tokens.clone()

        x_tokens = x_tokens[:, -self.in_context_positional_embedding_after.shape[1]:,
                   :] + self.in_context_positional_embedding_after[:, -x_tokens.shape[1]:, :]

        if not self.initialized:
            print('Reduced Token Size: ', x_tokens.shape)
            self.initialized = True
            # self.output_projection[str(future)].weight.requires_grad = False
            # self.output_projection[str(future)].bias.requires_grad = False

        limit = self.token_limit
        x_tokens = x_tokens[:, -limit:, :]
        mask = None
        if mask is not None:
            mask = mask[:, -limit:, -limit:]

        # x_tokens = x_tokens[:, -number_of_targets:, :]
        # x_tokens = x_tokens[:, -self.dec_in:, :]

        x_tokens = self.input_norm(x_tokens).to('cuda')
        # print(x_tokens.shape)
        N = x_tokens.shape[1]
        # self.dec_in = N
        if output_attention:
            x_tokens, attn = self.transformer_encoder(x_tokens, attn_mask=mask, N=N, output_attention=output_attention)
        else:
            x_tokens, _ = self.transformer_encoder(x_tokens, attn_mask=mask, N=N)  # B N*C d

        x_output = x_tokens[:, -number_of_targets:, :]
        x_tokens = self.output_norm(x_tokens)

        x_output = self.output_projection[str(future)](x_output)  # B C L_p

        x_output = x_output + token_mean[:, -x_output.shape[1]:, :]

        x_output = x_output.permute(0, 2, 1)

        if output_attention:
            return x_output, attn, x_tokens_orig, x_tokens_merged

        return x_output


def generate_indices(n):
    original_indices = list(range(n))
    shuffled_indices = original_indices[:]
    random.shuffle(shuffled_indices)

    restore_indices = [0] * n
    for original, shuffled in enumerate(shuffled_indices):
        restore_indices[shuffled] = original

    return shuffled_indices, restore_indices


def random_split_list(lst, max_parts=3):
    if len(lst) <= 1 or max_parts == 1:
        return [lst]

    n = len(lst)
    num_parts = random.randint(1, min(n, max_parts))

    cut_points = sorted(random.sample(range(1, n), num_parts - 1))
    cut_points = [0] + cut_points + [n]
    result = [lst[cut_points[i]:cut_points[i + 1]] for i in range(len(cut_points) - 1)]
    return result


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = ICTSP(lookback=configs.lookback,
                           output=configs.pred_len,
                           depth=configs.e_layers,
                           heads=configs.n_heads,
                           mlp_ratio=configs.mlp_ratio,
                           d_model=configs.d_model,
                           emb_init=0.01,
                           output_projection=True,
                           external_stride=configs.sampling_step,
                           external_context=True,
                           task_emb_dim=0,
                           n_channels=configs.enc_in,
                           channel_emb_dim=0,
                           partial_mask=False,
                           dropout=configs.dropout,
                           inter_series_latent_dim=0,
                           time_emb_dim=configs.time_emb_dim,
                           token_retriever_flag=configs.token_retriever_flag,
                           linear_warmup_steps=configs.linear_warmup_steps,
                           token_limit=configs.token_limit,
                           ICL_embedding=configs.ICL_embedding,
                           mask_length_for_comparison=configs.mask_length_for_comparison,
                           alpha=configs.alpha,
                           m=configs.m,
                           # dec_in = configs.dec_in,
                           wv=configs.wv,
                           kernel_size=configs.kernel_size,
                           geomattn_dropout=configs.geomattn_dropout,
                           d_ff=configs.d_ff, activation=configs.activation,
                           output_attention=configs.output_attention,
                           requires_grad=configs.requires_grad)
        self.fix_embedding = configs.fix_embedding
        self.independent = configs.sample_independent

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, output_attention=False):
        if self.fix_embedding:
            return self.model(x, None, None, x_mark_dec, output_attention=output_attention)
        else:
            if self.independent:
                output = []
                for c in range(x.shape[-1]):
                    output.append(self.model(x[:, :, [c]], None, None, x_mark_dec))
                return torch.cat(output, dim=-1)
            else:
                if self.training:
                    shuffle_indices, restore_indices = generate_indices(x.shape[-1])
                    split_indices = random_split_list(shuffle_indices, max_parts=1)
                    outputs = []
                    for ind in split_indices:
                        x_input = x[:, :, ind]
                        x_output = self.model(x_input, None, None, x_mark_dec, output_attention=output_attention)
                        outputs.append(x_output)
                    x_output = torch.cat(outputs, dim=-1)
                    output = x_output[:, :, restore_indices]
                else:
                    output = self.model(x, None, None, x_mark_dec, output_attention=output_attention)
                return output