import torch
from torch import nn 
from typing import Optional
from .modules.utils import move_to_device
from .modules.backbone import Transformer_Backbone
from .modules.vae_branch_modules import VAE_Branch
from .modules.mae_branch_modules import MAE_Branch
from .modules.seg_branch_modules import Segmentation_Branch


class VAE(nn.Module):
    def __init__(self,
                 input_dimension: int,
                 backbone_transformer_sequence_len: int,
                 vae_transformer_sequence_len: int,
                 backbone_n_features: int,
                 backbone_n_heads: int,
                 backbone_dropout_rate: float,
                 vae_n_features: int,
                 vae_n_heads: int,
                 vae_dropout_rate: float,
                 n_tokens: int, 
                 latent_vec_len: int,
                 latent_vec_output_shape: Optional[tuple]=None,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        """
        super(VAE, self).__init__()
        self.device = device
        self.dtype = dtype

        self.backbone = Transformer_Backbone(input_dimension=input_dimension,
                                             transformer_sequence_len=backbone_transformer_sequence_len,
                                             n_features=backbone_n_features,
                                             n_heads=backbone_n_heads,
                                             dropout_rate=backbone_dropout_rate,
                                             device=device,
                                             dtype=dtype)
        
        self.vae_branch = VAE_Branch(transformer_sequence_len=vae_transformer_sequence_len,
                                     n_features=vae_n_features,
                                     n_heads=vae_n_heads,
                                     dropout_rate=vae_dropout_rate,
                                     n_tokens=n_tokens,
                                     latent_vec_len=latent_vec_len,
                                     latent_vec_output_shape=latent_vec_output_shape,
                                     device=device,
                                     dtype=dtype)

    def forward(self,
                unmasked_input: torch.Tensor, 
                unmasked_token_indices: list[list[int]],
                input_positions: Optional[torch.Tensor]=None) -> list[torch.Tensor]:
        """
        """
        unmasked_input, input_positions = move_to_device(input1=unmasked_input,
                                                         input2=input_positions,
                                                         device=self.device,
                                                         dtype=self.dtype)
        backbone_output, embedded_unmasked_token_positions = \
                    self.backbone(unmasked_input=unmasked_input,
                                unmasked_token_indices=unmasked_token_indices,
                                input_positions=input_positions)
        
        z, mean, std, _ = \
            self.vae_branch(backbone_output=backbone_output,
                            embedded_position=embedded_unmasked_token_positions)
        
        return z, mean, std

class MAE(nn.Module):
    def __init__(self, 
                 backbone_transformer_sequence_len: int,
                 mae_transformer_sequence_len: int,
                 decoder_transformer_sequence_len: int,
                 backbone_n_features: int,
                 backbone_n_heads: int,
                 backbone_dropout_rate: float,
                 mae_enc_n_features: int,
                 mae_enc_n_heads: int,
                 mae_enc_dropout_rate: float,
                 mae_dec_n_features: int,
                 mae_dec_n_heads: int,
                 mae_dec_n_dropout_rate: int,
                 input_dimension: int,
                 mae_output_dimension: int,
                 mae_n_tokens: int,
                 n_points_per_token: int,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        """
        super(MAE, self).__init__()
        self.device = device
        self.dtype = dtype

        self.backbone = Transformer_Backbone(input_dimension=input_dimension,
                                             transformer_sequence_len=backbone_transformer_sequence_len,
                                             n_features=backbone_n_features,
                                             n_heads=backbone_n_heads,
                                             dropout_rate=backbone_dropout_rate)
        
        self.mae_branch = MAE_Branch(mae_transformer_sequence_len=mae_transformer_sequence_len,
                                     decoder_transformer_sequence_len=decoder_transformer_sequence_len,
                                     mae_enc_n_features=mae_enc_n_features,
                                     mae_enc_n_heads=mae_enc_n_heads,
                                     mae_enc_dropout_rate=mae_enc_dropout_rate,
                                     mae_dec_n_features=mae_dec_n_features,
                                     mae_dec_n_heads=mae_dec_n_heads,
                                     mae_dec_n_dropout_rate=mae_dec_n_dropout_rate,
                                     input_dimension=input_dimension,
                                     output_dimension=mae_output_dimension,
                                     n_tokens=mae_n_tokens,
                                     n_points_per_token=n_points_per_token,
                                     device=device,
                                     dtype=dtype,)
        
    def forward(self, 
                unmasked_input: torch.Tensor, 
                unmasked_token_indices: list[list[int]],
                masked_token_indices: list[list[int]],
                mae_cross_attention_input: Optional[torch.Tensor]=None,
                input_positions: Optional[torch.Tensor]=None) -> list[torch.Tensor]:
        """
        """
        unmasked_input, input_positions = move_to_device(input1=unmasked_input, 
                                                         input2=input_positions, 
                                                         device=self.device, 
                                                         dtype=self.dtype)
        backbone_output, embedded_unmasked_token_positions = \
            self.backbone(unmasked_input=unmasked_input,
                          unmasked_token_indices=unmasked_token_indices,
                          input_positions=input_positions)
        
        mae_output, _ =\
            self.mae_branch(unmasked_input=unmasked_input,
                            unmasked_token_indices=unmasked_token_indices,
                            masked_token_indices=masked_token_indices,
                            backbone_output=backbone_output,
                            embedded_unmasked_token_positions=embedded_unmasked_token_positions,
                            mae_cross_attention_input=mae_cross_attention_input,
                            input_positions=input_positions)

        return mae_output

class VAE_MAE_Seg_Model(nn.Module):
    def __init__(self,
                 input_dimension: int,
                 backbone_transformer_sequence_len: int,
                 vae_transformer_sequence_len: int,
                 mae_transformer_sequence_len: int,
                 decoder_transformer_sequence_len: int,
                 seg_transformer_sequence_len: int,
                 vae_n_features: int,
                 vae_n_heads: int,
                 vae_dropout_rate: float,
                 vae_n_tokens: int,
                 vae_latent_vec_len: int,
                 vae_latent_vec_output_shape: tuple,
                 backbone_n_features: int,
                 backbone_n_heads: int,
                 backbone_dropout_rate: float,
                 mae_enc_n_features: int,
                 mae_enc_n_heads: int,
                 mae_enc_dropout_rate: float,
                 mae_dec_n_features: int,
                 mae_dec_n_heads: int,
                 mae_dec_n_dropout_rate: int,
                 mae_output_dimension: int,
                 mae_n_tokens: int,
                 seg_enc_n_features: int,
                 seg_enc_n_heads: int,
                 seg_enc_dropout_rate: float,
                 seg_n_tokens: int,
                 n_points_per_token: int,
                 seg_output_dimension: int=44,
                 seg_softmax_dim: int=2,
                 vae: bool=True,
                 mae: bool=True,
                 seg: bool=True,
                 device: Optional[torch.device]=None,
                 dtype: Optional[torch.dtype]=None):
        """
        """
        super(VAE_MAE_Seg_Model, self).__init__()
        self.device = device
        self.dtype = dtype

        self.vae = vae
        self.mae = mae
        self.seg = seg

        self.vae_branch = None
        self.mae_branch = None
        self.seg_branch = None

        self.backbone = Transformer_Backbone(input_dimension=input_dimension,
                                             transformer_sequence_len=backbone_transformer_sequence_len,
                                             n_features=backbone_n_features,
                                             n_heads=backbone_n_heads,
                                             dropout_rate=backbone_dropout_rate,
                                             device=device,
                                             dtype=dtype)
        
        if self.vae:
            self.vae_branch = VAE_Branch(transformer_sequence_len=vae_transformer_sequence_len,
                                         n_features=vae_n_features,
                                         n_heads=vae_n_heads,
                                         dropout_rate=vae_dropout_rate,
                                         n_tokens=vae_n_tokens,
                                         latent_vec_len=vae_latent_vec_len,
                                         latent_vec_output_shape=vae_latent_vec_output_shape,
                                         device=device,
                                         dtype=dtype)
        
        if self.mae:
            self.mae_branch = MAE_Branch(mae_transformer_sequence_len=mae_transformer_sequence_len,
                                         decoder_transformer_sequence_len=decoder_transformer_sequence_len,
                                         mae_enc_n_features=mae_enc_n_features,
                                         mae_enc_n_heads=mae_enc_n_heads,
                                         mae_enc_dropout_rate=mae_enc_dropout_rate,
                                         mae_dec_n_features=mae_dec_n_features,
                                         mae_dec_n_heads=mae_dec_n_heads,
                                         mae_dec_n_dropout_rate=mae_dec_n_dropout_rate,
                                         input_dimension=input_dimension,
                                         output_dimension=mae_output_dimension,
                                         n_tokens=mae_n_tokens,
                                         n_points_per_token=n_points_per_token,
                                         device=device,
                                         dtype=dtype,)
        
        if self.seg:
            self.seg_branch = Segmentation_Branch(transformer_sequence_len=seg_transformer_sequence_len,
                                                  n_features=seg_enc_n_features,
                                                  n_heads=seg_enc_n_heads,
                                                  dropout_rate=seg_enc_dropout_rate,
                                                  n_tokens=seg_n_tokens,
                                                  n_points_per_token=n_points_per_token,
                                                  input_dimension=seg_enc_n_features,
                                                  output_dimension=seg_output_dimension,
                                                  softmax_dim=seg_softmax_dim,
                                                  device=device,
                                                  dtype=dtype)
        
    def forward(self,
                unmasked_input: torch.Tensor, 
                unmasked_token_indices: list[list[int]],
                masked_token_indices: list[list[int]],
                input_positions: Optional[torch.Tensor]=None,
                vae: bool=True,
                mae: bool=True,
                seg: bool=True) -> list[Optional[torch.Tensor]]:
        """
        """
        z = None
        mean = None
        std = None
        mae_output = None
        seg_output = None
        mae_cross_attention_input = None
        seg_cross_attention_input = None

        unmasked_input, input_positions = move_to_device(input1=unmasked_input, 
                                                         input2=input_positions, 
                                                         device=self.device, 
                                                         dtype=self.dtype)
        
        backbone_output, embedded_unmasked_token_positions = \
            self.backbone(unmasked_input=unmasked_input,
                          unmasked_token_indices=unmasked_token_indices,
                          input_positions=input_positions)
        
        if vae:
            z, mean, std, mae_cross_attention_input = \
                self.vae_branch(backbone_output=backbone_output,
                                embedded_position=embedded_unmasked_token_positions)
        
        if mae:
            mae_output, seg_cross_attention_input =\
                self.mae_branch(unmasked_input=unmasked_input,
                                unmasked_token_indices=unmasked_token_indices,
                                masked_token_indices=masked_token_indices,
                                backbone_output=backbone_output,
                                embedded_unmasked_token_positions=embedded_unmasked_token_positions,
                                mae_cross_attention_input=mae_cross_attention_input,
                                input_positions=input_positions)
        
        if seg:
            seg_output = self.seg_branch(unmasked_input=unmasked_input, 
                                         backbone_output=backbone_output,
                                         cross_attention_input=seg_cross_attention_input,
                                         embedded_position=embedded_unmasked_token_positions,
                                         seg_head_expansion_weight_input=mae_cross_attention_input)
        
        return z, mean, std, mae_output, seg_output
