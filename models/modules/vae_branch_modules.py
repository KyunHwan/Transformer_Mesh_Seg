import torch
from torch import nn 
from .utils import move_to_device
from .building_blocks import Activation, Transformer
from typing import Optional
from functools import reduce


class VAE_Branch_Linear_Encoder(nn.Module):
    """
    Description
    -----------
    A class used to pass input through a VAE branch linear encoder.

    Attributes
    ----------
    n_tokens: int
        Number of tokens from Transformer.
    n_features: int
        Number of features per token from Transformer.
    latent_vec_len: int
        Length of the encoded latent vector.
    latent_vec_output_shape: Optional[tuple]
        Shape of the (true) latent vector.
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    input: torch.Tensor
        Input in (B, N, n_features) format, where B is batch, N is the number of tokens, 
        and n_features is as described above.
    mean: torch.Tensor
        In (B, latent_vec_len) format. 
        This represents the mean vector of an inferred multi-variate Gaussian.
    std: torch.Tensor
        In (B, latent_vec_len) format. 
        This represents the std vector of an inferred multi-variate Gaussian. 
        
    Methods
    -------
    forward(self, input)
        Takes in input (B, N, n_features) and flattens it to (B, N * n_features).
        Then projects the vector to latent vector space.
    
    _reparameterization(self, mean: torch.Tensor, std: torch.Tensor)
        mean and std are in (B, latent_vec_len) shape.
        Returns an output in a (B, latent_vec_len) shape,
        where the output is mean + epsilon * std, where epsilon has the
        same shape with elements sampled from standard Gaussian.
    """
    def __init__(self,
                 n_tokens: int, 
                 n_features: int,
                 latent_vec_len: int,
                 latent_vec_output_shape: Optional[tuple]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        n_tokens: int
            Number of tokens from Transformer.
        n_features: int
            Number of features per token from Transformer.
        latent_vec_len: int
            Length of the encoded latent vector.
        latent_vec_output_shape: Optional[tuple]
            Shape of the (true) latent vector.
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        super(VAE_Branch_Linear_Encoder, self).__init__()

        if latent_vec_output_shape is not None:
            AssertionError(len(latent_vec_output_shape) == 2 and 
                           latent_vec_output_shape[0] * latent_vec_output_shape[1] == latent_vec_len)
        else:
            latent_vec_output_shape = (latent_vec_len, 1)

        self.device = device
        self.dtype=dtype
        self.latent_vec_len = latent_vec_len
        self.latent_vec_output_shape = latent_vec_output_shape

        self.gelu = Activation(activation=nn.functional.gelu)
        self.layernorm = nn.LayerNorm(normalized_shape=n_features, device=device, dtype=dtype)
        self.latent_projection = nn.ModuleList([
                                                nn.Linear(in_features=n_tokens * n_features, 
                                                          out_features=((n_tokens * n_features) + 2*latent_vec_len) // 2,
                                                          device=device, 
                                                          dtype=dtype),
                                                self.gelu,
                                                nn.Linear(in_features=((n_tokens * n_features) + 2*latent_vec_len) // 2, 
                                                          out_features=2*latent_vec_len,
                                                          device=device,
                                                          dtype=dtype),
                                                self.gelu,
                                                ])
        
    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        input: torch.Tensor
            Input in (B, N, n_features) format, where B is batch, N is the number of tokens, 
            and n_features is number of features per token from Transformer.
        
        Returns 
        -------
        list[torch.Tensor]
            Returns an output in a list of (B, *latent_vec_output_shape) torch.Tensor format.
            More specifically, outputs latent vector, mean vector, and std vector 
            in the aforementioned shape.
        """
        input, _ = move_to_device(input1=input, device=self.device, dtype=self.dtype)
        output = self.layernorm(input)
        
        output = output.view(output.shape[0], reduce((lambda x, y: x * y), output.shape[1:]))
        for layer in self.latent_projection:
            output = layer(output)

        mean, std = torch.split(output, self.latent_vec_len, dim=1)
        output = self._reparameterization(mean, std)

        if self.latent_vec_output_shape is not None:
            output = output.view(output.shape[0], *self.latent_vec_output_shape)
            mean = mean.view(mean.shape[0], *self.latent_vec_output_shape)
            std = std.view(std.shape[0], *self.latent_vec_output_shape)

        return output, mean, std
    
    def _reparameterization(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mean: torch.Tensor
            In (B, latent_vec_len) format. 
            This represents the mean vector of an inferred multi-variate Gaussian.
        std: torch.Tensor
            In (B, latent_vec_len) format. 
            This represents the std vector of an inferred multi-variate Gaussian. 

        Returns 
        -------
        torch.Tensor
            Returns an output in a (B, latent_vec_len) shape,
            where the output is mean + epsilon * std, where epsilon has the
            same shape with elements sampled from standard Gaussian.
        """
        epsilon = torch.randn_like(std, dtype=mean.dtype, device=self.device)
        z = mean + std*epsilon
        return z
    
class VAE_Branch(nn.Module):
    """
    Description
    -----------
    This is the VAE branch post-backbone.

    Attributes
    ----------
    transformer_sequence_len: int
        Number of transformer blocks.
    n_features: int
        Number of features per token for input to Transformer.
    n_heads: int
        Number of heads in multi-headed attention.
    dropout_rate: float
        Dropout rate in the Multi-hdeaded attention.
    n_tokens: int
        Number of tokens (ie. N in (B, N, C) shape) to and from transformer.
    latent_vec_len: int
        Length of the encoded latent vector.
    latent_vec_output_shape: Optional[tuple]
        Shape of the (true) latent vector.
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    backbone_output: torch.Tensor
        Output tensor from backbone in (B, N, C) shape.
    embedded_position: torch.Tensor
        Embedded position in (B, N, C) shape.

    Methods
    -------
    forward(self, 
            backbone_output: torch.Tensor,
            embedded_position: torch.Tensor,
            )
        Outputs latent vector, mean vector, std vector in (B, *latent_vec_output_shape) shape
        and output from VAE branch's transformer encoder in (B, N, C) shape.
    """
    def __init__(self,
                 transformer_sequence_len: int,
                 n_features: int,
                 n_heads: int,
                 dropout_rate: float,
                 n_tokens: int,
                 latent_vec_len: int,
                 latent_vec_output_shape: Optional[tuple]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        transformer_sequence_len: int
            Number of transformer blocks.
        n_features: int
            Number of features per token for input to Transformer.
        n_heads: int
            Number of heads in multi-headed attention.
        dropout_rate: float
            Dropout rate in the Multi-hdeaded attention.
        n_tokens: int
            Number of tokens (ie. N in (B, N, C) shape) to and from transformer.
        latent_vec_len: int
            Length of the encoded latent vector.
        latent_vec_output_shape: Optional[tuple]
            Shape of the (true) latent vector.
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        super(VAE_Branch, self).__init__()
        self.device = device
        self.dtype = dtype

        self.vae_transformer_encoder = Transformer(transformer_sequence_len=transformer_sequence_len,
                                                   n_features=n_features,
                                                   n_heads=n_heads,
                                                   dropout_rate=dropout_rate,
                                                   device=device,
                                                   dtype=dtype)

        self.vae_linear_encoder = VAE_Branch_Linear_Encoder(n_tokens=n_tokens,
                                                    n_features=n_features,
                                                    latent_vec_len=latent_vec_len,
                                                    latent_vec_output_shape=latent_vec_output_shape,
                                                    device=device,
                                                    dtype=dtype)
        
    def forward(self, 
                backbone_output: torch.Tensor,
                embedded_position: torch.Tensor,
                ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        backbone_output: torch.Tensor
            Output tensor from backbone in (B, N, C) shape.
        embedded_position: torch.Tensor
            Embedded position in (B, N, C) shape.

        Returns
        -------
        list[torch.Tensor]
            Outputs latent vector, mean vector, std vector in (B, *latent_vec_output_shape) shape
            and output from VAE branch's transformer encoder in (B, N, C) shape.
        """
        backbone_output, embedded_position = move_to_device(input1=backbone_output,
                                                            input2=embedded_position,
                                                            device=self.device,
                                                            dtype=self.dtype)
        vae_encoder_output, _, _ = \
                        self.vae_transformer_encoder(input=backbone_output, 
                                                     embedded_position=embedded_position)
        
        z, mean, std = self.vae_linear_encoder(vae_encoder_output)

        return z, mean, std, vae_encoder_output
