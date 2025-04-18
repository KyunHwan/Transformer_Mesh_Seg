import torch
from torch import nn 
from .utils import move_to_device, batch_index_select, batch_index_rearrange
from .building_blocks import Transformer, Activation, Linear_Interpolated_Expansion
from typing import Optional
from einops import repeat


class MAE_Branch_Transformer_Decoder(nn.Module):
    """
    Description
    -----------
    A class used to pass input(s) through a MAE branch Head.
    This follows the "Masked Autoencoders for Point Cloud Self-supervised Learning".

    Attributes
    ----------
    pos_input_dimension: int
        Input dimension, C, of the second input (B, N, C).
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
    input1: torch.Tensor
        Input in (B, N1, n_features) format, where B is batch, N1 is the number of tokens, 
        and n_features is as described above. 
        In the context of MAE framework, this should be unmasked tokens.
    unmasked_token_indices: list[list[int]]
        List of index list of unmasked tokens. 
        (ie. unmasked_token_indices[b][i] indicates 
        i^th unmasked token index in the full dataset in the b^th batch)
        len(unmasked_token_indices) = N1 from input1.
    masked_token_indices: list[list[int]]
        List of index list of masked tokens.
        (ie. masked_token_indices[b][i] indicates 
        i^th masked token index in the full dataset in the b^th batch)
        len(masked_token_indices) = N2.
    input2: Optional[torch.Tensor]
        Input in (B, N, C) format, where B is batch, N = N1 + N2 is the total number of tokens, 
        and C is the number of features per token before projection to n_features.
        (ex. positional input)
        This can be None.

    Methods
    -------
    forward(self, 
            input1: torch.Tensor, 
            unmasked_token_indices: list[list[int]],
            masked_token_indices: list[list[int]],
            input2: Optional[torch.Tensor]=None,
            ) -> torch.Tensor:
        Takes in input1 (B, N1, n_features) and input2 (B, N2, C). 
        Broadcasts masked_token to match the shape of input1. 
        Then concatenates input1 and masked_token, and then sorts the tokens wrt unmasked & masked token indices.
        Projects input2 to (B, N, n_features) and sums the two. Then passes the sum through a MAE branch Transformer.
    """
    def __init__(self, 
                 pos_input_dimension: int, 
                 transformer_sequence_len: int, 
                 n_features: int, 
                 n_heads: int, 
                 dropout_rate: float,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        pos_input_dimension: int
            Input dimension, C, of the second input (B, N, C).
        transformer_sequence_len: int
            Number of transformer blocks.
        n_features: int
            Number of features per token for input to Transformer.
        n_heads: int
            Number of heads in Multi-headed attention.
        dropout_rate: float
            Dropout rate in the Multi-headed attention.
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """

        super(MAE_Branch_Transformer_Decoder, self).__init__()
        self.device = device
        self.dtype = dtype
        self.masked_token = nn.Parameter(data=torch.ones(n_features, device=device, dtype=dtype))
        self.pos_embedding = nn.Linear(pos_input_dimension, n_features, device=device, dtype=dtype)
        self.transformer = Transformer(transformer_sequence_len=transformer_sequence_len,
                                       n_features=n_features,
                                       n_heads=n_heads,
                                       dropout_rate=dropout_rate, 
                                       device=device,
                                       dtype=dtype)
    
    def forward(self, 
                input: torch.Tensor, 
                unmasked_token_indices: list[list[int]],
                masked_token_indices: list[list[int]],
                input_position: Optional[torch.Tensor]=None,
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor
            Input in (B, N1, n_features) format, where B is batch, N1 is the number of tokens, 
            and n_features is as described above. In the context of MAE framework, this should be unmasked tokens.
        unmasked_token_indices: list[list[int]]
            List of index list of unmasked tokens. 
            (ie. unmasked_token_indices[b][i] indicates 
            i^th unmasked token index in the full dataset in the b^th batch)
            len(unmasked_token_indices) = N1 from input1.
        masked_token_indices: list[list[int]]
            List of index list of masked tokens.
            (ie. masked_token_indices[b][i] indicates 
            i^th masked token index in the full dataset in the b^th batch)
            len(masked_token_indices) = N2.
        input_position: Optional[torch.Tensor]
            Input in (B, N, C) format, where B is batch, N = N1 + N2 is the total number of tokens, 
            and C is the number of features per token before projection to n_features.
            (ex. Entire positional input)
            This can be None.

        Returns
        -------
        torch.Tensor
            Returns an output in a (B, N2, n_features) format.
        """

        input, input_position = move_to_device(input1=input, 
                                               input2=input_position, 
                                               device=self.device, 
                                               dtype=self.dtype)
        batch_masked_tokens = repeat(self.masked_token, 'c -> b n c', 
                                     b=len(masked_token_indices), 
                                     n=len(masked_token_indices[0]))
        
        intermediate_output = torch.cat((input, batch_masked_tokens), dim=1)
        num_batch, _, _ = intermediate_output.shape

        # Rearrange tokens to its original indices
        rearrange_batch_index = []
        for batch in range(num_batch):
            rearrange_batch_index.append(unmasked_token_indices[batch] + masked_token_indices[batch])
        output = batch_index_rearrange(input=intermediate_output, batch_indices=rearrange_batch_index)

        # Positional Embedding Summation
        embedded_position = None
        if input_position is not None:
            embedded_position = self.pos_embedding(input_position)

        output, _, _ = self.transformer(input=output, embedded_position=embedded_position)

        # Extract only the masked indices
        output = batch_index_select(input=output, batch_indices=masked_token_indices)

        return output

class MAE_Branch_Recon_Linear_Head(nn.Module):
    """
    Description
    -----------
    A class used to pass input(s) through a MAE branch recon head.
    The input should be in (B, N, n_features) format, where N is the number of tokens,
    n_features is the number of features per token, and B is the size of batch.
    This roughly follows the "Masked Autoencoders for Point Cloud Self-supervised Learning"
    and interpolation portion of the PointNet++ architecture.

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
        Takes in input (B, n_tokens, n_features). Then passes the input through a MAE branch recon head
        and outputs (B, n_tokens * n_points_per_token, output_dimension) tensor.
    """
    def __init__(self, 
                 output_dimension: int,
                 n_features: int, 
                 n_tokens: int,
                 n_points_per_token: int,
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
        super(MAE_Branch_Recon_Linear_Head, self).__init__()
        self.device = device
        self.dtype = dtype
        self.gelu = Activation(activation=nn.functional.gelu)

        # This is used to interpolate missing points from the existing tokens.
        self.linear_interpolated_expansion = Linear_Interpolated_Expansion(input1_dimension=n_tokens,
                                                                           output_dimension=n_tokens * n_points_per_token,
                                                                           input2_dimension=input2_dimension,
                                                                           device=device,
                                                                           dtype=dtype)

        # The input to this should be in (B, N, n_features) format.
        self.recon_head = nn.ModuleList([nn.Linear(n_features, n_features // 2, dtype=dtype, device=device),
                                        self.gelu,
                                        nn.Linear(n_features // 2, output_dimension, dtype=dtype, device=device)])
                                        
    def forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input1: torch.Tensor
            Input to the reconstruction head in (B, n_tokens, n_features) format, 
            where B is batch, N is the number of tokens,
            and n_features is the number of features per token.

        Returns
        -------
        torch.Tensor
            Returns an output in a (B, n_tokens * n_points_per_token, output_dimension) format, 
            where C is the output_dimension.
        """
        input1, input2 = move_to_device(input1=input1, input2=input2, device=self.device, dtype=self.dtype)
        output = self.linear_interpolated_expansion(input1, input2)
        for layer in self.recon_head:
            output = layer(output)
        return output
    
class MAE_Branch(nn.Module):
    def __init__(self, 
                 mae_transformer_sequence_len: int,
                 decoder_transformer_sequence_len: int,
                 mae_enc_n_features: int,
                 mae_enc_n_heads: int,
                 mae_enc_dropout_rate: float,
                 mae_dec_n_features: int,
                 mae_dec_n_heads: int,
                 mae_dec_n_dropout_rate: int,
                 input_dimension: int,
                 output_dimension: int,
                 n_tokens: int,
                 n_points_per_token: int,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        """
        super(MAE_Branch, self).__init__()
        self.device = device
        self.dtype = dtype

        self.mae_transformer_encoder = Transformer(transformer_sequence_len=mae_transformer_sequence_len,
                                                   n_features=mae_enc_n_features,
                                                   n_heads=mae_enc_n_heads,
                                                   dropout_rate=mae_enc_dropout_rate,
                                                   device=device,
                                                   dtype=dtype)

        self.mae_transformer_decoder = MAE_Branch_Transformer_Decoder(pos_input_dimension=input_dimension,
                                                                      transformer_sequence_len=decoder_transformer_sequence_len,
                                                                      n_features=mae_dec_n_features,
                                                                      n_heads=mae_dec_n_heads,
                                                                      dropout_rate=mae_dec_n_dropout_rate,
                                                                      device=device,
                                                                      dtype=dtype)
        
        self.recon_head = MAE_Branch_Recon_Linear_Head(output_dimension=output_dimension,
                                                       n_features=mae_dec_n_features, 
                                                       n_tokens=n_tokens,
                                                       n_points_per_token=n_points_per_token,
                                                       input2_dimension=None,
                                                       device=device,
                                                       dtype=dtype)

    def forward(self, 
                unmasked_input: torch.Tensor, 
                unmasked_token_indices: list[list[int]],
                masked_token_indices: list[list[int]],
                backbone_output: torch.Tensor,
                embedded_unmasked_token_positions: torch.Tensor,
                mae_cross_attention_input: Optional[torch.Tensor]=None,
                input_positions: Optional[torch.Tensor]=None) -> list[torch.Tensor]:
        """
        """
        _, input_positions = move_to_device(input1=unmasked_input, 
                                            input2=input_positions, 
                                            device=self.device, 
                                            dtype=self.dtype)
        
        seg_cross_attention_input, _, embedded_unmasked_token_positions = \
                        self.mae_transformer_encoder(input=backbone_output, 
                                                     cross_attention_input=mae_cross_attention_input,
                                                     embedded_position=embedded_unmasked_token_positions)
        
        output = self.mae_transformer_decoder(input=seg_cross_attention_input, 
                                              unmasked_token_indices=unmasked_token_indices,
                                              masked_token_indices=masked_token_indices,
                                              input_position=input_positions)

        output = self.recon_head(input1=output)

        return output, seg_cross_attention_input
    