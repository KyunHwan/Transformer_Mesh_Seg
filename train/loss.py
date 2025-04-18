import torch
from torch import nn
from .modules.loss_fncs import *
from .modules.schedulers import *
from typing import Optional


class VAE_Loss(nn.Module):
    """
    Description
    -----------
    VAE Loss (KL Divergence & MSE Loss) with integrated loss hyperparameter scheduler.
    The scheduler follows "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing".

    Attributes
    ----------
    total_data_num: int
        Total number of data.
    batch_size: int
        Batch size for batch training.
    epochs: int
        Number of epochs that the deep learning model will be trained for.
    prior_mean: float
        Prior Gaussian distribution's mean value.
    prior_var: float
        Prior Gaussian distribution's variance value.
    has_true_latent_vec: bool
        Whether or not true latent vec is available.
    cur_iter: int
        Current iteration number within the scope of the current epoch
    cur_epoch: int  
        Current training epoch that the model is going through.
    target_recon: torch.Tensor
        Ground truth reconstruction data in (B, *) format, where B is the batch dimension.
    pred_mean: torch.Tensor
        Predicted output of mean from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    pred_std: torch.Tensor
        Predicted output of standard deviation from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    target_reprojection_tensor: Optional[torch.Tensor]
        Reprojection tensor that when multiplied by pred_z via torch.bmm, 
        recovers original data in (B, *) format.
        Default=None
    pred_z: Optional[torch.Tensor]=None
        Predicted output of latent vector from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    target_z: Optional[torch.Tensor]=None
        Ground truth latent vector.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
        Default=None
    pred_recon: Optional[torch.Tensor]
        If deep learning model outputs torch.Tensor in (B, *) format instead of 
        only outputting pred_z.
        Default=None
    
    Methods
    -------
    forward(self, 
            cur_iter: int,
            cur_epoch: int,
            target_recon: torch.Tensor,
            pred_mean: torch.Tensor, 
            pred_std: torch.Tensor, 
            target_reprojection_tensor: Optional[torch.Tensor]=None,
            pred_z: Optional[torch.Tensor]=None,
            target_z: Optional[torch.Tensor]=None,
            pred_recon: Optional[torch.Tensor]=None
            )
        Calculates the loss for each iteration, 
        calculating the regularization term for the KL loss term every time.
    """

    def __init__(self,
                 total_data_num: int,
                 batch_size: int,
                 epochs: int,
                 prior_mean: float=0.0,
                 prior_var: float=1.0,
                 has_true_latent_vec: bool=False
                 ):
        """
        Parameters
        ----------
        total_data_num: int
            Total number of data.
        batch_size: int
            Batch size for batch training.
        epochs: int
            Number of epochs that the deep learning model will be trained for.
        prior_mean: float
            Prior Gaussian distribution's mean value.
        prior_var: float
            Prior Gaussian distribution's variance value.
        has_true_latent_vec: bool
            Whether or not true latent vec is available.
        """
        super(VAE_Loss, self).__init__()
        self.iter_per_epoch = iter_per_epoch(total_data_num=total_data_num, 
                                             batch_size=batch_size)
        self.total_iterations = total_training_iterations(total_data_num=total_data_num,
                                                          batch_size=batch_size,
                                                          epochs=epochs)
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.has_true_latent_vec = has_true_latent_vec

    def forward(self, 
                cur_iter: int,
                cur_epoch: int,
                target_recon: torch.Tensor,
                pred_mean: torch.Tensor, 
                pred_std: torch.Tensor, 
                target_reprojection_tensor: Optional[torch.Tensor]=None,
                pred_z: Optional[torch.Tensor]=None,
                target_z: Optional[torch.Tensor]=None,
                pred_recon: Optional[torch.Tensor]=None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        cur_iter: int
            Current iteration number within the scope of the current epoch
        cur_epoch: int  
            Current training epoch that the model is going through.
        target_recon: torch.Tensor
            Ground truth reconstruction data in (B, *) format, where B is the batch dimension.
        pred_mean: torch.Tensor
            Predicted output of mean from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        pred_std: torch.Tensor
            Predicted output of standard deviation from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        target_reprojection_tensor: Optional[torch.Tensor]
            Reprojection tensor that when multiplied by pred_z via torch.bmm, recovers original data.
            Default=None
        pred_z: Optional[torch.Tensor]=None
            Predicted output of latent vector from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        target_z: Optional[torch.Tensor]=None
            Ground truth latent vector.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
            Default=None
        pred_recon: Optional[torch.Tensor]
            If deep learning model outputs torch.Tensor in (B, *) format instead of 
            only outputting pred_z.
            Default=None
        
        Returns 
        -------
        torch.Tensor
            In (1, 1) format. This is the loss value.
        """
        loss = None
        iteration = cur_iteration(cur_iter=cur_iter,
                                  iter_per_epoch=self.iter_per_epoch,
                                  cur_epoch=cur_epoch)
        
        beta = cyclic_annealing_scheduler(iteration=iteration,
                                          total_iterations=self.total_iterations,)

        # KL Divergence Loss
        kl_loss = KL_divergence_loss(pred_mean=pred_mean,
                                     pred_std=pred_std,
                                     target_mean=self.prior_mean,
                                     target_var=self.prior_var,
                                     beta=0.5)

        # Reconstruction error
        if pred_recon is None and target_reprojection_tensor is not None:
            pred_recon = torch.bmm(target_reprojection_tensor, pred_z)
        recon_loss = mse_loss(pred=pred_recon, target=target_recon)

        loss = kl_loss + recon_loss
        # Latent vector regularization term
        if self.has_true_latent_vec and pred_z is not None and target_z is not None:
            loss += mse_loss(pred=pred_z, target=target_z, beta=beta)
            
        return loss

class MAE_Loss(nn.Module):
    """
    Description
    -----------
    Calculates MAE Loss (ie. reconstruction loss). 
    Either a MSE Loss or Chamfer loss, based on the loss version specified by the user.
    Density-aware chamfer loss follows "Density-aware Chamfer Distance 
    as a Comprehensive Metric for Point Cloud Completion" paper.

    Attributes
    ----------
    loss_version: str
        Either 'dcd', 'cd', or 'mse', each representing
        density-aware chamfer distance, chamfer distance, and mean squared distance,
        respectively. 
    pred_pcl: torch.Tensor
        Predicted point cloud in (B, N, C) format, where B is batch, 
        N is the number of points, and C is the number of features per point (ie. x, y, z in 3D).
    target_pcl: torch.Tensor
        Ground truth point cloud in (B, N, C) format, where B is batch, 
        N is the number of points, and C is the number of features per point (ie. x, y, z in 3D).
    reg: float
        Regularization term for the loss.

    Methods
    -------
    forward(self, 
            pred_pcl: torch.Tensor,
            target_pcl: torch.Tensor,
            reg: float=1.0,
            )
        Calculates loss for MAE architecture.
    """

    def __init__(self,
                 loss_version: str='dcd'):
        """
        Parameters
        ----------
        loss_version: str
            Either 'dcd', 'cd', or 'mse', each representing
            density-aware chamfer distance, chamfer distance, and mean squared distance,
            respectively. 
        """
        super(MAE_Loss, self).__init__()
        self.loss_version = loss_version

    def forward(self, 
                pred_pcl: torch.Tensor,
                target_pcl: torch.Tensor,
                reg: float=1.0,
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_pcl: torch.Tensor
            Predicted point cloud in (B, N, 3) format, where B is batch, 
            N is the number of points, and 3 is the number of axis per point (ie. x, y, z in 3D).
        target_pcl: torch.Tensor
            Ground truth point cloud in (B, N, 3) format, where B is batch, 
            N is the number of points, and 3 is the number of axis per point (ie. x, y, z in 3D).
        reg: float
            Regularization term for the loss.
        
        Returns 
        -------
        torch.Tensor
            In (1, 1) format. This is the loss value.
        """
        loss = None
        if self.loss_version == 'cd' or self.loss_version == 'dcd':
            loss = pcl_chamfer_dist(pred_pcl=pred_pcl, 
                                    target_pcl=target_pcl,
                                    reg=reg, 
                                    version=self.loss_version)
        else:
            loss = mse_loss(pred=pred_pcl, target=target_pcl, beta=1-reg)

        return loss
    
class Seg_Loss(nn.Module):
    """
    Description
    -----------
    Calculates loss for segmentation branch.

    Attributes
    ----------
    pred: torch.Tensor
        In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
        and C is the number of classes.
        Predicted B * N hot encoders output from a deep learning model.
    target: torch.Tensor
        In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
        and C is the number of classes.
        Target B * N hot encoders.
    reg: float
        Regularization term for the loss.
    label_smoothing: float
         A float in [0.0, 1.0]. 
         Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. 
         The targets become a mixture of the original ground truth and a uniform distribution 
         as described in "Rethinking the Inception Architecture for Computer Vision". 
         Default: 0.0025

    Methods
    -------
    forward(self, 
            pred: torch.Tensor,
            target: torch.Tensor,
            reg: float=1.0,
            label_smoothing: float=0.1
            )
        Calculates loss for segmentation branch.
    """

    def __init__(self,):
        """
        Parameters
        ----------
        None
        """
        super(Seg_Loss, self).__init__()

    def forward(self, 
                pred: torch.Tensor,
                target: torch.Tensor,
                reg: float=1.0,
                label_smoothing: float=0.0025
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred: torch.Tensor
            In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
            and C is the number of classes.
            Predicted B * N hot encoders output from a deep learning model.
        target: torch.Tensor
            In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
            and C is the number of classes.
            Target B * N hot encoders.
        reg: float
            Regularization term for the loss.
        label_smoothing: float
            A float in [0.0, 1.0]. 
            Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. 
            The targets become a mixture of the original ground truth and a uniform distribution 
            as described in "Rethinking the Inception Architecture for Computer Vision". 
            Default: 0.0025
        
        Returns
        -------
        torch.Tensor
            In (1, 1) format. It's the loss value.
        """
        loss = cross_entropy_loss(pred=pred,
                                  target=target,
                                  reg=reg,
                                  label_smoothing=label_smoothing)
        return loss
    
class VAE_MAE_Seg_Loss(nn.Module):
    """
    Description
    -----------
    VAE Loss (KL Divergence & MSE Loss) with integrated loss hyperparameter scheduler.
    The scheduler follows "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing".

    Attributes
    ----------
    total_data_num: int
        Total number of data.
    batch_size: int
        Batch size for batch training.
    epochs: int
        Number of epochs that the deep learning model will be trained for.
    vae_prior_mean: float
        Prior Gaussian distribution's mean value.
    vae_prior_var: float
        Prior Gaussian distribution's variance value.
    vae_has_true_latent_vec: bool
        Whether or not true latent vec is available.
    mae_loss_version: str
        Either 'dcd', 'cd', or 'mse', each representing
        density-aware chamfer distance, chamfer distance, and mean squared distance,
        respectively. 
    vae: bool
        Whether or not VAE loss should be computed.
    mae: bool
        Whether or not MAE loss should be computed.
    seg: bool
        Whether or not segmentation loss should be computed.
    mae_pred: Optional[torch.Tensor]
        Predicted point cloud in (B, N, C') format, where B is batch, 
        N is the number of points, and C' is the number of features per point (ie. x, y, z in 3D).
    mae_target: Optional[torch.Tensor]
        Ground truth point cloud in (B, N, C') format, where B is batch, 
        N is the number of points, and C' is the number of features per point (ie. x, y, z in 3D).
    seg_pred: Optional[torch.Tensor]
        In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
        and C is the number of classes.
        Predicted B * N hot encoders output from a deep learning model.
    seg_target: Optional[torch.Tensor]
        In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
        and C is the number of classes.
        Target B * N hot encoders.
    vae_cur_iter: Optional[int]
        Current iteration number within the scope of the current epoch.
    vae_cur_epoch: Optional[int]
        Current training epoch that the model is going through.
    vae_target_reprojection_tensor: Optional[torch.Tensor]
        Reprojection tensor that when multiplied by vae_pred_z via torch.bmm, 
        recovers original data in (B, *) format.
        Default=None
    vae_target_recon: Optional[torch.Tensor]
        Ground truth reconstruction data in (B, *) format, where B is the batch dimension.
    vae_pred_mean: Optional[torch.Tensor]
        Predicted output of mean from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    vae_pred_std: Optional[torch.Tensor] 
        Predicted output of standard deviation from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    vae_pred_z: Optional[torch.Tensor]
        Predicted output of latent vector from a deep learning model.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
    vae_target_z: Optional[torch.Tensor]
        Ground truth latent vector.
        In (B, *') format, where B is the batch dimension and *' is any dimensions.
        Default=None
    seg_reg: float
        Regularization term for the Segmentation loss.
    seg_label_smoothing: float
         A float in [0.0, 1.0]. 
        Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. 
        The targets become a mixture of the original ground truth and a uniform distribution 
        as described in "Rethinking the Inception Architecture for Computer Vision". 
        Default: 0.0025
    mae_reg: float
        Regularization term for the MAE loss.
    vae: bool
        Whether or not VAE loss should be computed.
    mae: bool
        Whether or not MAE loss should be computed.
    seg: bool
        Whether or not segmentation loss should be computed.
    
    Methods
    -------
    forward(self,
            mae_pred: Optional[torch.Tensor]=None,
            mae_target: Optional[torch.Tensor]=None,
            seg_pred: Optional[torch.Tensor]=None,
            seg_target: Optional[torch.Tensor]=None,
            vae_cur_iter: Optional[int]=None,
            vae_cur_epoch: Optional[int]=None,
            vae_target_reprojection_tensor: Optional[torch.Tensor]=None,
            vae_target_recon: Optional[torch.Tensor]=None,
            vae_pred_mean: Optional[torch.Tensor]=None, 
            vae_pred_std: Optional[torch.Tensor]=None, 
            vae_pred_z: Optional[torch.Tensor]=None,
            vae_target_z: Optional[torch.Tensor]=None,
            seg_reg: float=1.0,
            seg_label_smoothing: float=0.0025,
            mae_reg: float=1.0,
            vae: bool=True,
            mae: bool=True,
            seg: bool=True)
        Calculates the loss for each iteration for 
        VAE, MAE, and/or Segmentation
    """
    def __init__(self,
                 total_data_num: int,
                 batch_size: int,
                 epochs: int,
                 vae_prior_mean: float=0.0,
                 vae_prior_var: float=1.0,
                 vae_has_true_latent_vec: bool=False,
                 mae_loss_version: str='dcd',
                 vae: bool=True,
                 mae: bool=True,
                 seg: bool=True):
        """
        Parameters
        ----------
        total_data_num: int
            Total number of data.
        batch_size: int
            Batch size for batch training.
        epochs: int
            Number of epochs that the deep learning model will be trained for.
        vae_prior_mean: float
            Prior Gaussian distribution's mean value.
        vae_prior_var: float
            Prior Gaussian distribution's variance value.
        vae_has_true_latent_vec: bool
            Whether or not true latent vec is available.
        mae_loss_version: str
            Either 'dcd', 'cd', or 'mse', each representing
            density-aware chamfer distance, chamfer distance, and mean squared distance,
            respectively. 
        vae: bool
            Whether or not VAE loss should be computed.
        mae: bool
            Whether or not MAE loss should be computed.
        seg: bool
            Whether or not segmentation loss should be computed.
        """
        super(VAE_MAE_Seg_Loss, self).__init__()

        self.vae = vae
        self.mae = mae
        self.seg = seg
        self.vae_loss = None
        self.mae_loss = None
        self.seg_loss = None

        if self.vae:
            self.vae_loss = VAE_Loss(prior_mean=vae_prior_mean,
                                    prior_var=vae_prior_var,
                                    total_data_num=total_data_num,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    has_true_latent_vec=vae_has_true_latent_vec)
        
        if self.mae:
            self.mae_loss = MAE_Loss(loss_version=mae_loss_version)

        if self.seg:
            self.seg_loss = Seg_Loss()

    def forward(self,
                mae_pred: Optional[torch.Tensor]=None,
                mae_target: Optional[torch.Tensor]=None,
                seg_pred: Optional[torch.Tensor]=None,
                seg_target: Optional[torch.Tensor]=None,
                vae_cur_iter: Optional[int]=None,
                vae_cur_epoch: Optional[int]=None,
                vae_target_reprojection_tensor: Optional[torch.Tensor]=None,
                vae_target_recon: Optional[torch.Tensor]=None,
                vae_pred_mean: Optional[torch.Tensor]=None, 
                vae_pred_std: Optional[torch.Tensor]=None, 
                vae_pred_z: Optional[torch.Tensor]=None,
                vae_target_z: Optional[torch.Tensor]=None,
                seg_reg: float=1.0,
                seg_label_smoothing: float=0.0025,
                mae_reg: float=1.0,
                vae: bool=True,
                mae: bool=True,
                seg: bool=True) -> torch.Tensor:
        """
        Parameters
        ----------
        mae_pred: Optional[torch.Tensor]
            Predicted point cloud in (B, N, C') format, where B is batch, 
            N is the number of points, and C' is the number of features per point (ie. x, y, z in 3D).
        mae_target: Optional[torch.Tensor]
            Ground truth point cloud in (B, N, C') format, where B is batch, 
            N is the number of points, and C' is the number of features per point (ie. x, y, z in 3D).
        seg_pred: Optional[torch.Tensor]
            In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
            and C is the number of classes.
            Predicted B * N hot encoders output from a deep learning model.
        seg_target: Optional[torch.Tensor]
            In (B, N, C) format, where B is the batch dimension and N is the number of points in the point cloud
            and C is the number of classes.
            Target B * N hot encoders.
        vae_cur_iter: Optional[int]
            Current iteration number within the scope of the current epoch.
        vae_cur_epoch: Optional[int]
            Current training epoch that the model is going through.
        vae_target_reprojection_tensor: Optional[torch.Tensor]
            Reprojection tensor that when multiplied by vae_pred_z via torch.bmm, 
            recovers original data in (B, *) format.
            Default=None
        vae_target_recon: Optional[torch.Tensor]
            Ground truth reconstruction data in (B, *) format, where B is the batch dimension.
        vae_pred_mean: Optional[torch.Tensor]
            Predicted output of mean from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        vae_pred_std: Optional[torch.Tensor] 
            Predicted output of standard deviation from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        vae_pred_z: Optional[torch.Tensor]
            Predicted output of latent vector from a deep learning model.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
        vae_target_z: Optional[torch.Tensor]
            Ground truth latent vector.
            In (B, *') format, where B is the batch dimension and *' is any dimensions.
            Default=None
        seg_reg: float
            Regularization term for the Segmentation loss.
        seg_label_smoothing: float
            A float in [0.0, 1.0]. 
            Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. 
            The targets become a mixture of the original ground truth and a uniform distribution 
            as described in "Rethinking the Inception Architecture for Computer Vision". 
            Default: 0.0025
        mae_reg: float
            Regularization term for the MAE loss.
        vae: bool
            Whether or not VAE loss should be computed.
        mae: bool
            Whether or not MAE loss should be computed.
        seg: bool
            Whether or not segmentation loss should be computed.

        Returns
        -------
        torch.Tensor
            In (1, 1) format. It's the loss value.
        """
        loss = 0.0
        if vae:
            loss = loss + self.vae_loss(cur_iter=vae_cur_iter,
                                        cur_epoch=vae_cur_epoch,
                                        target_reprojection_tensor=vae_target_reprojection_tensor,
                                        target_recon=vae_target_recon,
                                        pred_mean=vae_pred_mean,
                                        pred_std=vae_pred_std,
                                        pred_z=vae_pred_z,
                                        target_z=vae_target_z,)
            
        if mae:
            loss = loss + self.mae_loss(pred_pcl=mae_pred,
                                        target_pcl=mae_target,
                                        reg=mae_reg,)
        
        if seg:
            loss = loss + self.seg_loss(pred=seg_pred, 
                                        target=seg_target,
                                        reg=seg_reg,
                                        label_smoothing=seg_label_smoothing,)
        
        return loss
    