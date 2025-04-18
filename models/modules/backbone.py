import torch
from torch import nn
from .building_blocks import Input_Embedding, Transformer
from .utils import move_to_device, batch_index_select
from typing import Optional


class Transformer_Backbone(nn.Module):
    """
    Description
    -----------
    A class used to pass input through a transformer based backbone.

    Attributes
    ----------
    input_dimension: int
        Input dimension of the input tensor (ie. C in (B, N, C) tensor,
        where B is the batch, N is the number of points and C is for features).
    transformer_sequence_len: int
        Number of transformer blocks.
    n_features: int
        Number of features per token for input to Transformer.
    n_heads: int
        Number of heads in multi-headed attention.
    dropout_rate: float
        Dropout rate in the Multi-hdeaded attention.
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    unmasked_input: torch.Tensor
        Raw input to the backbone in (B, N, input_dimension) format.
    unmasked_token_indices: Optional[list[list[int]]]
        List of index list to select for each batch from input_positions. 
        len(unmasked_token_indices) == B.
    input_positions: Optional[torch.Tensor]
        Input position tensor in (B, N', imput_dimension) format, where N' >= N.
        If unmasked_token_indices is None, N' has to equal N.
        
    Methods
    -------
    forward(self, 
            unmasked_input: torch.Tensor, 
            unmasked_token_indices: list[list[int]],
            input_positions: Optional[torch.Tensor]=None)
        Takes in unmasked_input, which is in (B, N, input_dimension) format and optionally
        input_positions, which is in (B, N', input_dimension) format, where N' >= N along with
        unmasked_token_indices with which to select a subtensor of input_positions. 
        If input_positions is not None, then both unmasked_input and selected subset of input_positions
        go through separate linear embedding and then are passed through transformer blocks. 
    """
    def __init__(self,
                 input_dimension: int,
                 transformer_sequence_len: int,
                 n_features: int,
                 n_heads: int,
                 dropout_rate: float,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        input_dimension: int
            Input dimension of the input tensor (ie. C in (B, N, C) tensor,
            where B is the batch, N is the number of points and C is for features).
        transformer_sequence_len: int
            Number of transformer blocks.
        n_features: int
            Number of features per token for input to Transformer.
        n_heads: int
            Number of heads in multi-headed attention.
        dropout_rate: float
            Dropout rate in the Multi-hdeaded attention.
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        super(Transformer_Backbone, self).__init__()
        self.device = device
        self.dtype = dtype
        self.input_embedding = Input_Embedding(input1_dimension=input_dimension, 
                                               output_dimension=n_features,
                                               device=device, 
                                               dtype=dtype)
        
        self.enc_position_embedding = Input_Embedding(input1_dimension=input_dimension, 
                                                      output_dimension=n_features,
                                                      device=device, 
                                                      dtype=dtype)
        
        self.backbone_transformer = Transformer(transformer_sequence_len=transformer_sequence_len, 
                                                n_features=n_features, 
                                                n_heads=n_heads,
                                                dropout_rate=dropout_rate, 
                                                device=device, 
                                                dtype=dtype)
        
    def forward(self, 
                unmasked_input: torch.Tensor, 
                unmasked_token_indices: Optional[list[list[int]]]=None,
                input_positions: Optional[torch.Tensor]=None) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        unmasked_input: torch.Tensor
            Raw input to the backbone in (B, N, input_dimension) format.
        unmasked_token_indices: Optional[list[list[int]]]
            List of index list to select for each batch from input_positions. 
            len(unmasked_token_indices) == B.
        input_positions: Optional[torch.Tensor]
            Input position tensor in (B, N', imput_dimension) format, where N' >= N.
            If unmasked_token_indices is None, N' has to equal N.

        Returns
        -------
        list[torch.Tensor]
            Returns a list of torch tensors composed of output from the backbone
            and position embedding, both in (B, N, n_features) format. 
        """
        unmasked_input, input_positions = move_to_device(input1=unmasked_input, 
                                                         input2=input_positions, 
                                                         device=self.device, 
                                                         dtype=self.dtype)
        
        embedded_unmasked_token_positions = None
        if unmasked_token_indices is not None:
            unmasked_token_positions = batch_index_select(input=input_positions, 
                                                          batch_indices=unmasked_token_indices)
            embedded_unmasked_token_positions = self.enc_position_embedding(unmasked_token_positions)
        else:
            embedded_unmasked_token_positions = self.enc_position_embedding(input_positions)
        
        embedded_unmasked_input = self.input_embedding(unmasked_input)
        backbone_output, _, embedded_unmasked_token_positions = \
                        self.backbone_transformer(input=embedded_unmasked_input,
                                                  embedded_position=embedded_unmasked_token_positions)
    
        return backbone_output, embedded_unmasked_token_positions
    