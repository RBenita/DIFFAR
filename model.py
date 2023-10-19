import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d
ConvTranspose1d = nn.ConvTranspose1d

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  
        dims = torch.arange(64).unsqueeze(0)                   
        table = steps * 10.0**(dims * 4.0 / 63.0)        
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class MRB(nn.Module):
    def __init__(self, channels_in, channels_out, kernels, dilation):
        super().__init__()
        self.n_kernels = len(kernels)
        layers = []
        for k in kernels:
            layers.append(Conv1d(channels_in, channels_out, k, padding=get_padding(k, dilation), dilation=dilation))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y / self.n_kernels


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = MRB(residual_channels, 2 * residual_channels, [3,5,7], dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = MRB(1, 2 * residual_channels, [3,5,7], dilation=dilation)  
        self.conditioner_phoneme = MRB(1, 1, [3,5,7], dilation=dilation)
        self.conditioner_energy = MRB(1, 1, [3,5,7], dilation=dilation)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, phoneme_conditioner, energy_conditioner ):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)  
        conidtioned_phoneme_embedding = self.conditioner_phoneme(phoneme_conditioner)
        conidtioned_energy_embedding = self.conditioner_energy(energy_conditioner)  
        y = x + diffusion_step 
        y_step_0 = self.dilated_conv(y) + conditioner
        y_step_1 = y_step_0 + conidtioned_phoneme_embedding
        y = y_step_1 + conidtioned_energy_embedding
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class DiffAR(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.residual_layers = nn.ModuleList([
                ResidualBlock(params.residual_channels, 2**(i % params.dilation_cycle_length))
                for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, conditioner, diffusion_step, phoneme_conditioner, energy_conditioner):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, conditioner, diffusion_step, phoneme_conditioner, energy_conditioner)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x