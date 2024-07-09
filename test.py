import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln1 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.ln2 = nn.LayerNorm(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.ln1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ProsodyEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, num_blocks, layers_per_block, kernel_size=3):
        super(ProsodyEncoder, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size // 2)
        self.initial_ln = nn.LayerNorm(hidden_size)
        self.initial_relu = nn.ReLU()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                *[ResidualBlock(hidden_size, hidden_size, kernel_size) for _ in range(layers_per_block)]
            ) for _ in range(num_blocks)
        ])

        self.attentive_pooling = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.final_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.final_ln = nn.LayerNorm(hidden_size)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.initial_relu(x)

        for block in self.blocks:
            x = block(x)

        x, _ = self.attentive_pooling(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))
        x = x.transpose(0, 1).transpose(1, 2)

        x = self.final_conv(x)
        x = self.final_ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.final_relu(x)

        return x


# Define the model
input_channels = 128  # Number of input channels (mel-spectrogram length)
hidden_size = 192  # Hidden size for the conv layers
num_blocks = 4  # Number of residual blocks
layers_per_block = 12  # Number of conv layers per block
model = ProsodyEncoder(input_channels, hidden_size, num_blocks, layers_per_block)

# Check the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params / 1e6:.2f}M")

# Example input tensor with batch size 1 and sequence length 500
x = torch.randn(1, input_channels, 500)
output = model(x)
print(output.shape)


# TEST2


class AttentivePooling1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(AttentivePooling1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multihead Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Linear layers to transform input_dim to hidden_dim
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Transform input x to hidden_dim using linear layers
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Permute to fit the attention layer (seq_len, batch, embed_dim)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Apply multi-head attention
        attn_output, _ = self.attention(query, key, value)

        # Permute back to (batch, seq_len, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output


class ProsodyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_residual_blocks, num_conv_layers_per_block, pooling_hidden_size,
                 num_heads, output_dim):
        super(ProsodyEncoder, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
                        nn.ReLU(),
                        nn.LayerNorm(hidden_size)
                    ) for _ in range(num_conv_layers_per_block)
                ]
            ) for _ in range(num_residual_blocks)
        ])

        self.attentive_pooling = AttentivePooling1D(hidden_size, pooling_hidden_size, num_heads)

        # Final Linear Layer to match the output size
        self.final_linear = nn.Linear(pooling_hidden_size, output_dim)

    def forward(self, x):
        # Initial Conv
        x = x.permute(0, 2, 1)  # (N, 128, input_dim) -> (N, input_dim, 128)
        x = self.initial_conv(x)

        # Residual Blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x += residual  # Add residual connection

        x = x.permute(0, 2, 1)  # (N, input_dim, 128) -> (N, 128, input_dim)

        # Attentive Pooling
        x = self.attentive_pooling(x)

        # Final Linear Layer
        x = self.final_linear(x)  # (N, 128, pooling_hidden_size) -> (N, 128, output_dim)

        return x


# Prosody Encoder Parameters
mel_spectrogram_length = 128
hidden_size = 192
num_residual_blocks = 4
num_conv_layers_per_block = 12
pooling_hidden_size = 768
num_heads = 4

# Create the prosody encoder
prosody_encoder = ProsodyEncoder(input_dim=192, hidden_size=hidden_size, num_residual_blocks=num_residual_blocks,
                                 num_conv_layers_per_block=num_conv_layers_per_block,
                                 pooling_hidden_size=pooling_hidden_size, num_heads=num_heads, output_dim=output_dim)

# Example input: batch of mel-spectrograms
input_mel_spectrograms = torch.randn(16, 128, 192)  # (N, 128, input_dim)

# Get the output embeddings
prosody_output_embeddings = prosody_encoder(input_mel_spectrograms)
print(f"Prosody Encoder Output Shape: {prosody_output_embeddings.shape}")  # Expected output shape: (16, 128, 768)
