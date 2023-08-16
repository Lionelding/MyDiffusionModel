import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    def __init__(self, beta) -> None:
        self.beta = beta
        self.step = 0
        
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, new_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, new_weight)
            
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
        
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step = self.step + 1
            return 
        self.update_model_average(ema_model, model)
        self.step = self.step + 1
        
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channel=None, residual=False) -> None:
        super().__init__()
        self.residual = residual
        if not mid_channel:
            mid_channel = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channel),
            nn.GELU(),
            nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim, 
                out_channels
            )
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        """
        Args:
            channels int: channel dimension
            size int: Image resolution
        """
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) # [Batch, H, W, C]
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size) #[Batch, C, H, W]
