import torch
from torch import nn
from typing import Optional
from einops import rearrange
from .utils import move_to_device
    
class Activation(nn.Module):
    """
    Description
    -----------
    Class that implements torch activation function.

    Attributes
    ----------
    activation:
        This is an activation function (Should be torch.nn.functional type)
    dim: Optional[int]
        Dimension to which the activation function should apply. If None, 
        the activation function applies elementwise.
    input: torch.Tensor
        This is an input to the activation function.

    Methods
    -------
    forward(self, input)
        Passes the input through the activation function.
    """
    def __init__(self, activation, dim: Optional[int]=None):
        """
        Parameters
        ----------
        activation:
            This is an activation function (Should be torch.nn.functional type)
        dim: Optional[int]
            Dimension to which the activation function should apply. If None, 
            the activation function applies elementwise.
        """
        super(Activation, self).__init__()
        self.activation = None
        if dim is not None:
            self.activation = activation(dim=dim)
        else:
            self.activation = activation
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input:
            Input to the activation function.

        Returns
        -------
        torch.Tensor
            Returns a activated input
        """
        return self.activation(input)

class Input_Embedding(nn.Module):
    """
    Description
    -----------
    A class used to embed input(s)' feature dimension using linear projection, 
    where input(s) are in (B, N, C1) and (B, N, C2) format.

    Attributes
    ----------
    input_dimension: int
        Input dimension of each token (ie. C1 in the Description).
    output_dimension: int
        Output dimension of each token.
    input2_dimension: Optional[int]
        Input dimension of each token for input2 (ie. C2 in the Description).
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    input1: torch.Tensor
        Input in (B, N, C1) format, where B is batch, N is the number of data points, 
        and C1 is the number of features per point. C1 is equal to input_dimension.
    input2: Optional[torch.Tensor]
        Input in (B, N, C2) format, where B is batch, N is the number of data points, 
        and C2 is the number of features per point. C2 is equal to input2_dimension.
        This can be None.
    
    Methods
    -------
    forward(self, input1, input2=None)
        Takes in input1 and input2, in (B, N, C1) and (B, N, C2) format.
        (Ex. C1 = C2 = k x k, where k is the number of points in a patch for input1 
        and k only consists of the patch's center position for input2.) 
        Projects the feature dimension C1 & C2 onto output dimension for both inputs
        and outputs the sum of linearly projected input1 and input2.
        If input2 is None, only input1 is processed.
    """
    def __init__(self, 
                 input1_dimension: int, 
                 output_dimension: int, 
                 input2_dimension: Optional[int]=None, 
                 device: Optional[torch.device]=None, 
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        input_dimesnion: int
            Input dimension of each token (ie. C1 in the Description).
        output_dimesnion: int
            Output dimension of each token.
        input2_dimension: Optional[int]
            Input dimension of each token for input2 (ie. C2 in the Description).
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        super(Input_Embedding, self).__init__()
        self.device = device
        self.dtype = dtype
        self.input_embedding1 = nn.Linear(input1_dimension, 
                                          output_dimension, 
                                          device=device, 
                                          dtype=dtype)
        
        self.input_embedding2 = None
        if input2_dimension is not None:
            self.input_embedding2 = nn.Linear(input2_dimension, 
                                              output_dimension, 
                                              device=device, 
                                              dtype=dtype)

    def forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input1: torch.Tensor
            Input in (B, N, C1) format, where B is batch, N is the number of data points, 
            and C1 is the number of features per point. C1 is equal to input_dimension.
            (Ex. N could be number of patches in point cloud and C1 could be number of features per patch,
            and C1 = k x k, where k is the number of points per patch.)
        input2: Optional[torch.Tensor]=None
            This could be None, in which case only input1 is projected. 
            Input in (B, N, C2) format, where B is batch, N is the number of data points, 
            and C2 is the number of features per point. C2 is equal to input2_dimension.
            (Ex. N could be number of patches in point cloud and C2 could be number of features per patch,
            and C2 = k x k, where k is the number of points per patch.)
        
        Returns
        -------
        torch.Tensor
            Returns a projected output in (B, N, D) format, where D is equal to output_dimension.
        """
        output = None
        input1, input2 = move_to_device(input1=input1, 
                                        input2=input2, 
                                        device=self.device, 
                                        dtype=self.dtype)

        if input2 is None or self.input_embedding2 is None:
            output = self.input_embedding1(input1)
        else:
            output = self.input_embedding1(input1) + self.input_embedding2(input2)
        return output

class Transformer_Block(nn.Module):
    """
    Description
    -----------
    A class used to pass input through a Transformer Block.

    Attributes
    ----------
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
    input: torch.Tensor
        Input in (B, N, C) format, where B is batch, N is the number of tokens, 
        and C is the number of features per token.
    cross_attention_input: Optional[torch.Tensor]
        Cross attention input in (B, N, C) format, where B is batch, N is the number of tokens,
        and C is the number of features per token.
    embedded_position: Optional[torch.Tensor]
        Embedded position in (B, N, C) format, where B is batch, N is the number of tokens, 
        and C is the number of features per token.
        This is added to the input, but not to the cross attention input.

    Methods
    -------
    forward(self, 
            input: torch.Tensor, 
            cross_attention_input: Optional[torch.Tensor]=None,
            embedded_position: Optional[torch.Tensor]=None)
        Takes in inputs in (B, N, C) format and passes it through a standard Transformer Block.
        Addss embedded position onto input and then passes input and cross attention input through
        the Transformer block, where cross attention input is passed as key and value.  
        Returns output of Transformer in (B, N, C) format, a cross attention input in (B, N, C) format,
        and embedded position in put in (B, N, C) format.
    """
    def __init__(self, 
                 n_features: int, 
                 n_heads: int, 
                 dropout_rate: float, 
                 device: Optional[torch.device]=None, 
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
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
        super(Transformer_Block, self).__init__()
        self.device = device
        self.dtype = dtype
        self.activation = Activation(activation=nn.functional.gelu)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=n_features, device=device, dtype=dtype)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=n_features, device=device, dtype=dtype)
        self.mha = nn.MultiheadAttention(embed_dim=n_features, 
                                         num_heads=n_heads,
                                         dropout=dropout_rate,
                                         device=device,
                                         dtype=dtype)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=n_features, device=device, dtype=dtype)
        self.nonlinear_mlp_layer = nn.ModuleList([nn.Linear(n_features, 
                                                            n_features, 
                                                            device=device, 
                                                            dtype=dtype),
                                                  self.activation,
                                                  nn.Linear(n_features, 
                                                            n_features, 
                                                            device=device, 
                                                            dtype=dtype),
                                                  self.activation])

    def forward(self, 
                input: torch.Tensor, 
                cross_attention_input: Optional[torch.Tensor]=None,
                embedded_position: Optional[torch.Tensor]=None) -> list[Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input: torch.Tensor
            Input in (B, N, C) format, where B is batch, N is the number of tokens, 
            and C is the number of features per token.
        cross_attention_input: Optional[torch.Tensor]
            Cross attention input in (B, N, C) format, where B is batch, N is the number of tokens,
            and C is the number of features per token.
        embedded_position: Optional[torch.Tensor]
            Embedded position in (B, N, C) format, where B is batch, N is the number of tokens, 
            and C is the number of features per token.
            This is added to the input, but not to the cross attention input.

        Returns
        -------
        torch.Tensor
            Takes in inputs in (B, N, C) format and passes it through a standard Transformer Block.
            Addss embedded position onto input and then passes input and cross attention input through
            the Transformer block, where cross attention input is passed as key and value.  
            Returns output of Transformer in (B, N, C) format, a cross attention input in (B, N, C) format,
            and embedded position in put in (B, N, C) format.
        """
        output = None
        input, cross_attention_input = move_to_device(input1=input, 
                                                      input2=cross_attention_input, 
                                                      device=self.device, 
                                                      dtype=self.dtype)
        input, embedded_position = move_to_device(input1=input, 
                                                  input2=embedded_position, 
                                                  device=self.device, 
                                                  dtype=self.dtype)
        if embedded_position is not None:
            input = input + embedded_position

        output = self.layer_norm1(input)

        if cross_attention_input is not None:
            cross_attention_input = self.layer_norm2(cross_attention_input)
            output, _ = self.mha(query=output, 
                                 key=cross_attention_input, 
                                 value=cross_attention_input, 
                                 need_weights=False)
        else:
            output, _ = self.mha(query=output, 
                                 key=output, 
                                 value=output, 
                                 need_weights=False)

        intermediate_output = input + output
        output = self.layer_norm3(intermediate_output)

        for layer in self.nonlinear_mlp_layer:
            output = layer(output)

        output = output + intermediate_output

        return output, cross_attention_input, embedded_position

class Transformer(nn.Module):
    """
    Description
    -----------
    A class used to pass input through a Transformer.

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
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    input: torch.Tensor
        Input in (B, N, C) format, where B is batch, N is the number of tokens, 
        and C is the number of features per token.
    cross_attention_input: Optional[torch.Tensor]
        Cross attention input in (B, N, C) format, where B is batch, N is the number of tokens,
        and C is the number of features per token.
    embedded_position: Optional[torch.Tensor]
        Embedded position in (B, N, C) format, where B is batch, N is the number of tokens, 
        and C is the number of features per token.
        This is added to the input, but not to the cross attention input.
        
    Methods
    -------
    forward(self, 
            input: torch.Tensor, 
            cross_attention_input: Optional[torch.Tensor]=None,
            embedded_position: Optional[torch.Tensor]=None)
        Takes in inputs in (B, N, C) format and passes it through a standard Transformer Block.
        Addss embedded position onto input and then passes input and cross attention input through
        the Transformer block, where cross attention input is passed as key and value.  
        Returns output of Transformer in (B, N, C) format, a cross attention input in (B, N, C) format,
        and embedded position in put in (B, N, C) format.
    """
    def __init__(self, 
                 transformer_sequence_len: int, 
                 n_features: int, 
                 n_heads: int,
                 dropout_rate: float, 
                 device: Optional[torch.device]=None, 
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        transformer_sequence_len: int
            Number of transformer block.
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
        super(Transformer, self).__init__()
        self.device = device
        self.dtype = dtype
        self.transformers = nn.ModuleList([Transformer_Block(n_features=n_features, 
                                                              n_heads=n_heads, 
                                                              dropout_rate=dropout_rate,
                                                              device=device,
                                                              dtype=dtype) 
                                                              for i in range(transformer_sequence_len)])
    def forward(self, 
                input: torch.Tensor, 
                cross_attention_input: Optional[torch.Tensor]=None,
                embedded_position: Optional[torch.Tensor]=None) -> list[Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input: torch.Tensor
            Input in (B, N, C) format, where B is batch, N is the number of tokens, 
            and C is the number of features per token.
        cross_attention_input: Optional[torch.Tensor]
            Cross attention input in (B, N, C) format, where B is batch, N is the number of tokens,
            and C is the number of features per token.
        embedded_position: Optional[torch.Tensor]
            Embedded position in (B, N, C) format, where B is batch, N is the number of tokens, 
            and C is the number of features per token.
            This is added to the input, but not to the cross attention input.  
        Returns
        -------
        torch.Tensor
            Takes in inputs in (B, N, C) format and passes it through a standard Transformer Block.
            Addss embedded position onto input and then passes input and cross attention input through
            the Transformer block, where cross attention input is passed as key and value.  
            Returns output of Transformer in (B, N, C) format, a cross attention input in (B, N, C) format,
            and embedded position in put in (B, N, C) format.
        """
        input, cross_attention_input = move_to_device(input1=input, 
                                                      input2=cross_attention_input, 
                                                      device=self.device, 
                                                      dtype=self.dtype)
        output, embedded_position = move_to_device(input1=input, 
                                                  input2=embedded_position, 
                                                  device=self.device, 
                                                  dtype=self.dtype)

        for transformer_block in self.transformers:
            output, cross_attention_input, embedded_position = \
                transformer_block(input=output, 
                                  cross_attention_input=cross_attention_input, 
                                  embedded_position=embedded_position,)


        return output, cross_attention_input, embedded_position

class Linear_Interpolated_Expansion(nn.Module):
    """
    Description
    -----------
    A class used to pass input in (B, N, C) format, where N is the number of tokens,
    C is the number of features per token, and B is the size of a batch.
    This outputs a tensor in (B, N', C) format, where N' is the expanded/contracted version of N. 
    This is used to control the number of points.

    If input2 is given, it is used to generate an interpolation mapping from input2 that
    maps input1 to the desired interpolated/expanded output.
    (ie. (B, N, C2) goes through FCN to output (B, N, N'). Then this output is multiplied by
         rearranged input1 in (B, C1, N) format to output (B, C1, N'). Then this output is
         rearranged to become (B, N', C1).)

    This roughly follows the interpolation portion of the PointNet++ architecture.

    Attributes
    ----------
    input1_dimension: int
        Input dimension, N, of (B, N, C1).
    output_dimension: int
        Output dimension, N', of (B, N', C).
    input2_dimension: Optional[int]
        Input feature dimension, C2, of (B, N, C2)
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    input1: torch.Tensor
        Input to the reconstruction head in (B, input1_dimension, C) format.
        (ex. B is batch, input1_dimension is the number of tokens,
        and C is the number of features per token.)
    input2: Optional[torch.Tensor]
        Input to the reconstruction head in (B, input1_dimension, input2_dimension) format.
        (ex. B is batch, input1_dimension is the number of tokens,
        and input2_dimension is the number of features per token for input2.)
    
    Methods
    -------
    forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor]=None) -> torch.Tensor:
        Takes in input (B, N, C). Then outputs an expanded/interpolated version, (B, N', C).
    """
    def __init__(self,
                 input1_dimension: int,
                 output_dimension: int,
                 input2_dimension: Optional[int]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        Parameters
        ----------
        input1_dimension: int
            Input dimension, N, of (B, N, C1).
        output_dimension: int
            Output dimension, N', of (B, N', C).
        input2_dimension: Optional[int]
            Input feature dimension, C2, of (B, N, C2)
        device: Optional[torch.device]
            Device to which the model parameters are mapped (ex. cpu or gpu).
        dtype: Optional[torch.dtype]
            Data type of the model parameters.
        """
        
        super(Linear_Interpolated_Expansion, self).__init__()
        self.device = device
        self.dtype = dtype
        self.input2_dimension = input2_dimension

        self.softmax = Activation(activation=nn.Softmax, dim=1)
        if input2_dimension is None:
            self.linear_expansion_interpolation = nn.Linear(input1_dimension, 
                                                            output_dimension, 
                                                            device=device, 
                                                            dtype=dtype)
        else:
            self.linear_expansion_interpolation = nn.ModuleList([nn.Linear(input2_dimension,
                                                                           output_dimension, 
                                                                           device=device, 
                                                                           dtype=dtype),
                                                                self.softmax])

    def forward(self, input1: torch.Tensor, input2: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Parameters
        ----------
        input1: torch.Tensor
            Input to the reconstruction head in (B, input1_dimension, C) format.
            (ex. B is batch, input1_dimension is the number of tokens,
            and C is the number of features per token.)
        input2: Optional[torch.Tensor]
            Input to the reconstruction head in (B, input1_dimension, input2_dimension) format.
            (ex. B is batch, input1_dimension is the number of tokens,
            and input2_dimension is the number of features per token for input2.)

        Returns
        -------
        torch.Tensor
            Returns an output in a (B, output_dimension, C) format.
        """
        output = None
        input1, input2 = move_to_device(input1=input1, 
                                        input2=input2, 
                                        device=self.device, 
                                        dtype=self.dtype)
        input1 = rearrange(input1, 'b n c -> b c n')

        if input2 is None or self.input2_dimension is None: 
            output = self.linear_expansion_interpolation(input1)    
        else:
            interp_w = input2
            for layer in self.linear_expansion_interpolation:
                interp_w = layer(interp_w)
            output = torch.bmm(input1, interp_w)

        output = rearrange(output, 'b c n -> b n c')
        return output
    