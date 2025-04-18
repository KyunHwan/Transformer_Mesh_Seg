import torch
from torch import nn 
from .utils import move_to_device
from .building_blocks import Activation, Transformer, Linear_Interpolated_Expansion
from typing import Optional


class Segmentation_Branch_Linear_Head(nn.Module):
    """
    Description
    -----------
    A class used to pass input through a segmentation head.
    This roughly follows the "Masked Autoencoders for Point Cloud Self-supervised Learning" and
    PointNet++ for interpolation portion.

    Attributes
    ----------
    output_dimension: int
        Output dimension, C, of (B, N, C).
    n_features: int
        Number of features per token as output from Transformer.
    n_tokens: int
        Number of tokens 
    n_points_per_token: int
        Number of points that each token should be used to reconstruct.
        n_tokens * n_points_per_token should equal the number of points this head is trying to reconstruct.
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
        
    Methods
    -------
    forward(self, input: torch.Tensor) -> torch.Tensor:
        Takes in input (B, n_tokens, n_features) and outputs (B, n_tokens * n_points_per_token, C) tensor,
        where C is passed through softmax function.
    """

    def __init__(self, 
                 n_features: int, 
                 n_tokens: int,
                 n_points_per_token: int,
                 output_dimension: int=44,
                 softmax_dim: int=2,
                 input2_dimension: Optional[int]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        output_dimension: int
            Output dimension, C, of (B, N, C).
        n_features: int
            Number of features per token as output from Transformer.
        n_tokens: int
            Number of tokens 
        n_points_per_token: int
            Number of points that each token should be used to reconstruct.
            n_tokens * n_points_per_token should equal the number of points this head is trying to reconstruct.
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        super(Segmentation_Branch_Linear_Head, self).__init__()
        self.device = device
        self.dtype = dtype
        self.intermediate_output_dim = max(n_features // 2, (n_features + output_dimension) // 2)

        self.gelu = Activation(activation=nn.functional.gelu)
        self.softmax = Activation(activation=nn.Softmax, dim=softmax_dim)
        self.linear_interpolated_expansion = Linear_Interpolated_Expansion(input1_dimension=n_tokens,
                                                                           output_dimension=n_tokens * n_points_per_token,
                                                                           input2_dimension=input2_dimension,
                                                                           device=device,
                                                                           dtype=dtype)
        self.segmentation_head = nn.ModuleList([nn.Linear(n_features, 
                                                         self.intermediate_output_dim, 
                                                         device=device, 
                                                         dtype=dtype),
                                                self.gelu,
                                                nn.Linear(self.intermediate_output_dim, 
                                                          output_dimension, 
                                                          device=device, 
                                                          dtype=dtype),
                                                self.softmax])
                                                
    def forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input1: torch.Tensor
            Input to the segmentation head in (B, n_tokens, n_features) format.

        Returns
        -------
        torch.Tensor
            Returns an output in a (B, n_tokens * n_points_per_token, output_dimension) format.
        """
        input1, input2 = move_to_device(input1=input1, input2=input2, device=self.device, dtype=self.dtype)

        output = self.linear_interpolated_expansion(input1, input2)
        for layer in self.segmentation_head:
            output = layer(output)

        return output
    
class Segmentation_Branch(nn.Module):
    def __init__(self,
                 transformer_sequence_len: int,
                 n_features: int,
                 n_heads: int,
                 dropout_rate: float,
                 n_tokens: int,
                 n_points_per_token: int,
                 input_dimension: int,
                 output_dimension: int=44,
                 softmax_dim: int=2,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        """
        super(Segmentation_Branch, self).__init__()
        self.device = device
        self.dtype = dtype

        self.seg_transformer_encoder = Transformer(transformer_sequence_len=transformer_sequence_len,
                                                   n_features=n_features,
                                                   n_heads=n_heads,
                                                   dropout_rate=dropout_rate,
                                                   device=device,
                                                   dtype=dtype)
        
        self.seg_head = Segmentation_Branch_Linear_Head(n_features=n_features, 
                                                        n_tokens=n_tokens,
                                                        n_points_per_token=n_points_per_token,
                                                        output_dimension=output_dimension,
                                                        softmax_dim=softmax_dim,
                                                        input2_dimension=input_dimension,
                                                        device=device,
                                                        dtype=dtype)
    def forward(self, 
                unmasked_input: torch.Tensor,
                backbone_output: torch.Tensor,
                cross_attention_input: torch.Tensor,
                embedded_position: torch.Tensor,
                seg_head_expansion_weight_input: torch.Tensor):
        """
        """
        backbone_output, _ = move_to_device(input1=backbone_output, 
                                            input2=unmasked_input, 
                                            device=self.device, 
                                            dtype=self.dtype)
        backbone_output, seg_head_expansion_weight_input = move_to_device(input1=backbone_output,
                                                                          input2=seg_head_expansion_weight_input,
                                                                          device=self.device,
                                                                          dtype=self.dtype)
        
        backbone_output, cross_attention_input = move_to_device(input1=backbone_output,
                                                                input2=cross_attention_input,
                                                                device=self.device,
                                                                dtype=self.dtype)
        backbone_output, embedded_position = move_to_device(input1=backbone_output,
                                                            input2=embedded_position,
                                                            device=self.device,
                                                            dtype=self.dtype)
        seg_output, _, _ = \
            self.seg_transformer_encoder(input=backbone_output, 
                                         cross_attention_input=cross_attention_input,
                                         embedded_position=embedded_position)

        seg_output = self.seg_head(input1=seg_output, input2=seg_head_expansion_weight_input)

        return seg_output
    