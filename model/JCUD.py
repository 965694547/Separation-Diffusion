import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DiffusionEmbedding(nn.Module):
    """ Diffusion Step Embedding """

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class JCUDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, model_config):
        super(JCUDiscriminator, self).__init__()

        n_input_channels = model_config["Encoder"].conv1d.out_channels // 2
        residual_channels = model_config["step_embed"]["diffusion_step_embed_dim_in"]
        n_layer = model_config["discriminator_params"]["n_layer"]
        n_uncond_layer = model_config["discriminator_params"]["n_uncond_layer"]
        n_cond_layer = model_config["discriminator_params"]["n_cond_layer"]
        n_channels = model_config["discriminator_params"]["n_channels"]
        kernel_sizes = model_config["discriminator_params"]["kernel_sizes"]
        strides = model_config["discriminator_params"]["strides"]

        self.encoder = Encoder(
            in_channels=model_config["Encoder"].conv1d.in_channels,
            out_channels=model_config["Encoder"].conv1d.out_channels,
            kernel_size=model_config["Encoder"].conv1d.kernel_size[0],
            stride=model_config["Encoder"].conv1d.stride[0],
            padding=model_config["Encoder"].conv1d.padding[0],
        )
        self.input_projection = LinearNorm(2 * n_input_channels, 2 * n_input_channels)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, n_channels[n_layer-1]),
        )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1] if i != 0 else 2 * n_input_channels,
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer)
            ]
        )
        self.uncond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_uncond_layer)
            ]
        )
        self.cond_conv_block = nn.ModuleList(
            [
                ConvNorm(
                    n_channels[i-1],
                    n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    dilation=1,
                )
                for i in range(n_layer, n_layer + n_cond_layer)
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x_ts, x_t_prevs, t):
        """
        x_ts -- [B, T]
        x_t_prevs -- [B, T]
        t -- [B]
        """
        x = self.encoder(torch.cat([x_t_prevs, x_ts], dim=-1).unsqueeze(1))
        x = self.input_projection(x).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t.squeeze(-1))).unsqueeze(-1)

        cond_feats = []
        uncond_feats = []
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
            uncond_feats.append(x)

        x_cond = (x + diffusion_step)
        x_uncond = x

        for layer in self.cond_conv_block:
            x_cond = F.leaky_relu(layer(x_cond), 0.2)
            cond_feats.append(x_cond)

        for layer in self.uncond_conv_block:
            x_uncond = F.leaky_relu(layer(x_uncond), 0.2)
            uncond_feats.append(x_uncond)
        return cond_feats, uncond_feats