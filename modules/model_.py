import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
# ============================== DEFINING INDIVIDUAL POSITIONING WITHIN DIFFERENT BATCHES ============================
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding used in Transformers.
    Adds position-dependent signals to input embeddings so the model 
    can use order information because it would normally be order agnostic.

    Parameters
    ----------
    d_model : int
        Dimensionality of the embeddings.
    max_len : int, default=1000
        Maximum sequence length supported.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        # Creating empty encoding matrix
        pe = torch.zeros(max_len, d_model)
        # Checking how many positions to add
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Building the frequency scale for the positional encoding!
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model))
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Here adding the positional encoding
        """
        x = x + self.pe[:x.size(0)]
        return x
        
class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom Transformer encoder layer with:
    - Multihead self-attention
    - Position-wise feedforward network
    - Dropout + LayerNorm

    Parameters
    ----------
    d_model : int
        Dimensionality of embeddings.
    nhead : int
        Number of attention heads.
    dim_feedforward : int, default=512
        Hidden size in the feedforward sub-layer.
    dropout : float, default=0.1
        Dropout probability.
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        ## Multihead self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        ## Feedforward netowrk
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        ## Layernorm and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Transformer encoder forward pass useful to implement multihead self-attention,
        feedforward network, dropout and layer norm to input

        Parameters
        ----------
        src : torch.Tensor
            Input of shape (seq_len, batch, d_model).
        src_mask : torch.Tensor, optional
            Attention mask.
        src_key_padding_mask : torch.Tensor, optional
            Padding mask.

        Returns
        -------
        src : torch.Tensor
            Encoded output (seq_len, batch, d_model).
        attn_weights : torch.Tensor
            Attention weights (batch, nhead, seq_len, seq_len).
        """
        ## This was useful at the start to store attention weights!
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn_weights

class VariationalTimeSeriesTransformer(nn.Module):
    """
    Transformer-based Variational Encoder Decoder for time-series forecasting.

    Combines:
    - Transformer encoder layers for temporal dependencies
    - Latent Gaussian distribution (mu, logvar)
    - Reparameterisation trick for sampling
    - Sequential decoder to reconstruct/forecast future steps

    Parameters
    ----------
    input_features : int, default=11
        Number of input features per timestep.
    d_model : int, default=128
        Transformer embedding dimension.
    latent_dim : int, default=64
        Dimension of latent Gaussian space.
    nhead : int, default=8
        Number of attention heads.
    num_layers : int, default=6
        Number of stacked encoder layers.
    dim_feedforward : int, default=512
        Hidden size of feedforward layers.
    dropout : float, default=0.2
        Dropout probability.
    forecast_steps : int, default=1
        Number of timesteps to forecast.
    output_features : int, default=11
        Number of output features per forecast step.
    """
    def __init__(
        self,
        input_features: int = 11,
        d_model: int = 128,
        latent_dim: int = 64, 
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        forecast_steps: int = 1,
        output_features: int = 11
    ):
        super().__init__()

        ## Input projection to d_model

        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        ## Encoder layers implementing customer encoder structure

        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Latent mean and log variance projection
        self.mean_layer = nn.Linear(d_model, latent_dim)
        self.logvar_layer = nn.Linear(d_model, latent_dim)

        # Sequential Decoder maps from latent space → forecast
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward), ## first linear layer
            nn.ReLU(), ## adding non linearity
            nn.Linear(dim_feedforward, output_features * forecast_steps), ## second linear layer
        )

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.forecast_steps = forecast_steps
        self.output_features = output_features

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample a latent vector z from a Gaussian distribution 
        parameterized by mean and log-variance, using the 
        reparameterization trick.

        This makes the sampling differentiable by expressing it as:
            z = mu + std * eps
        where eps ~ N(0, I).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian, shape (batch, latent_dim).
        logvar : torch.Tensor
            Log-variance of the latent Gaussian, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Latent sample z, shape (batch, latent_dim).
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src: torch.Tensor):
        """
        Forward pass of the VariationalTimeSeriesTransformer.

        Steps:
        1. Project input features to d_model dimension and add positional encodings.
        2. Pass through stacked Transformer encoder layers, collecting attention weights (not used)
        3. Use the final time step representation to compute latent mean (mu) and log-variance (logvar).
        4. Sample latent vector z using the reparameterization trick.
        5. Decode z into forecasted outputs.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (seq_len, batch, input_features).

        Returns
        -------
        recon : torch.Tensor
            Reconstructed forecast of shape (batch, forecast_steps, output_features).
        mu : torch.Tensor
            Mean of latent Gaussian, shape (batch, latent_dim).
        logvar : torch.Tensor
            Log-variance of latent Gaussian, shape (batch, latent_dim).
        attentions : list of torch.Tensor
            Attention weight matrices from each encoder layer.
            Each of shape (batch, nhead, seq_len, seq_len).
        """
        x = self.input_proj(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        attentions = []
        for layer in self.encoder_layers:
          x, attn_weights = layer(x)
          attentions.append(attn_weights)
        enc_out = x


        ## Take last time-step of encoder - sumarise all previous info
        last = enc_out[-1, :, :]  # (batch, d_model)

        ## Latent space
        mu = self.mean_layer(last)      # (batch, latent_dim)
        logvar = self.logvar_layer(last)  # (batch, latent_dim)

        ## Sample from latent distribution
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # Decode
        recon = self.decoder(z)  # (batch, forecast_steps * output_features)

        if self.forecast_steps > 1:
            recon = recon.view(-1, self.forecast_steps, self.output_features)

        return recon, mu, logvar, attentions

class LatentDiscriminator(nn.Module):

    """
    Discriminator network for adversarial regularization of the latent space.
    This network takes a latent vector z as input and outputs a probability 
    that z was sampled from the prior distribution N(0, I) rather than the 
    encoder
    """
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output: probability z is from N(0, I)
        )
        

    def forward(self, z):
        """
        Forward pass of latent discriminator
        """
        return self.net(z)
