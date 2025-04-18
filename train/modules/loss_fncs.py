import torch
from torch import nn
from pytorch3d.loss import chamfer_distance
from einops import repeat


def KL_divergence_loss(pred_mean: torch.Tensor, 
                       pred_std: torch.Tensor,
                       target_mean: float=0.0, 
                       target_var: float=1.0,
                       beta: float=1.0,) -> torch.Tensor:
    """
    Description
    -----------
    Calculates KL Divergence Loss between the pred and the target, weighted using the beta regularization factor.
    This assumes predicted tensor components are all iid Gaussian, with the same prior as Gaussian.

    Parameters
    ----------
    pred_mean: torch.Tensor
        Predicted output of mean from a deep learning model.
        In (B, *) format, where B is the batch dimension and * is any dimensions.
    pred_std: torch.Tensor
        Predicted output of standard deviation from a deep learning model.
        In (B, *) format, where B is the batch dimension and * is any dimensions.
    target_mean: float
        Prior Gaussian distribution's mean value.
    target_var: float
        Prior Gaussian distribution's variance value.
    beta: float
        regularization factor for the loss.
        
    Returns
    -------
    torch.Tensor
        In (1, 1) format. This is the loss value.
    """
    device = pred_mean.device
    dtype = pred_mean.dtype

    prior_mean = torch.full(size=pred_mean.shape, fill_value=target_mean, dtype=dtype, device=device)
    prior_var = torch.full(size=pred_mean.shape, fill_value=target_var, dtype=dtype, device=device)
    safe_division_factor = torch.full(size=pred_mean.shape, fill_value=1e-8, dtype=dtype, device=device)
    ones = torch.ones(size=pred_mean.shape, dtype=dtype, device=device)
    reg = beta / (2 * pred_mean.shape[0])

    pred_var = torch.square(pred_std)

    # Averaged by batch number
    loss = torch.sum(
                      (torch.log(prior_var) - torch.log(pred_var) + \
                       torch.div((pred_var + torch.square(pred_mean - prior_mean)), prior_var + safe_division_factor) - \
                       ones)
                    ) * reg

    return loss
    
def mse_loss(pred: torch.Tensor, target: torch.Tensor, beta: float=0.0,) -> torch.Tensor:
    """
    Description
    -----------
    Calculates MSE Loss between the pred and the target, weighted using the (1 - beta) regularization factor.
    This assumes predicted tensor components are all iid Gaussian, with the same prior as Gaussian.

    Parameters
    ----------
    pred: torch.Tensor
        Predicted output from a deep learning model. 
        In (B, *) format, where B is the batch dimension and * is the tensor dimensions.
    target: torch.Tensor
        Target against which the predicted output is compared.
        In (B, *) format, where B is the batch dimension and * is the tensor dimensions.
    beta: float
        hyperparameter for the loss. The loss will be scaled by (1 - beta)
        
    Returns
    -------
    torch.Tensor
        In (1, 1) format. It's the loss value.
    """
    device = pred.device
    dtype = pred.dtype
    mse_loss_fnc = nn.MSELoss()
    if target.get_device() != pred.get_device() or target.dtype != pred.dtype:
        target = target.to(device=device, dtype=dtype)

    if beta != 0.0:
        reg = 1.0 - beta
        loss = reg * mse_loss_fnc(pred, target)
    else:
        loss = mse_loss_fnc(pred, target)

    return loss

def _argmin_freq(s1: torch.Tensor, s2: torch.Tensor, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]: 
    """
    Description
    -----------
    This returns for each point in s1, a corresponding point in s2 that minimizes the L2 value, 
    as well as the frequency of the corresponding points in s2 that minimizes L2 value for each point in s1.
    These are both in (B, N) format where B is the batch dimension and N is the number of points. 
    Follows the "Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion" paper.

    Parameters
    ----------
    s1: torch.Tensor
        point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    s2: torch.Tensor
        point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    device: torch.device
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: torch.dtype
        Data type of the model parameters.

    Returns
    -------
    list[torch.Tensor]
        Two torch.Tensors in (B, N) format.
        This returns for each point in s1, a corresponding point in s2 that minimizes the L2 value, 
        as well as the frequency of the corresponding points in s2 that minimizes L2 value for each point in s1.
    """
    num_batch, n_points, _ = s1.shape
    argmin = torch.empty(size=(num_batch, n_points)).type(torch.LongTensor).to(device)
    min_freq = torch.empty(size=(num_batch, n_points), device=device)

    for n_point in range(n_points):
        n_point_batch_coords = s1[:, n_point, :]
        broadcasted = repeat(n_point_batch_coords, 'b c -> b n c', n=n_points)
        dist = torch.linalg.norm(broadcasted - s2, dim=2)
        argmin[:, n_point] = torch.argmin(dist, dim=1)
    
    for batch in range(num_batch):
        min_freq[batch, :] = torch.index_select(input=argmin[batch, :].bincount(), 
                                                dim=0, 
                                                index=argmin[batch, :])

    return argmin, min_freq

def _one_way_chamfer_dist(s1: torch.Tensor, 
                          s2: torch.Tensor, 
                          wrt_s1_argmin: torch.LongTensor, 
                          wrt_s1_min_freq: torch.Tensor, 
                          alpha: float,
                          device: torch.device, 
                          dtype: torch.dtype) -> torch.Tensor:
    """
    Description
    -----------
    This returns the loss value of one sided density-aware chamfer distance, 
    normalized by the number of samples in batch.
    Follows the "Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion" paper.

    Parameters
    ----------
    s1: torch.Tensor
        point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    s2: torch.Tensor
        point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    wrt_s1_argmin: torch.LongTensor
        For each point in s1, a corresponding point in s2 that minimizes the L2 value.
        In (B, N) format.
    wrt_s1_min_freq: torch.Tensor
        Frequency of the corresponding points in s2 that minimizes L2 value for each point in s1.
        In (B, N) format.
    alpha: float
        Hyperparameter in front of the L2 distance as mentioned in the paper. 
        Default is 50.0
    device: torch.device
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: torch.dtype
        Data type of the model parameters.

    Returns
    -------
    list[torch.Tensor]
        In (1, 1) format. It's the loss value of one sided density-aware chamfer distance,
        normalized by the number of samples in batch.
    """
    num_batch, n_points, _ = s1.shape
    wrt_s1_argmin = wrt_s1_argmin.to(device=device)
    wrt_s1_min_freq = wrt_s1_min_freq.to(device=device, dtype=dtype)

    ones = torch.ones(n_points, device=device, dtype=dtype)
    loss = 0.0
    for batch in range(num_batch):
        loss += (torch.sum(
            ones - (1/wrt_s1_min_freq[batch,:]) * 
            torch.exp(
                    -alpha * torch.linalg.norm(s1[batch, :, :] - 
                                                torch.index_select(input=s2[batch, :, :], 
                                                                    dim=0, 
                                                                    index=wrt_s1_argmin[batch, :]),
                                                dim=1)))) * (1/n_points)
        
    return loss / num_batch

def density_aware_chamfer_dist(pred_pcl: torch.Tensor, target_pcl: torch.Tensor, alpha: float=50.0,) -> torch.Tensor:
    """
    Description
    -----------
    Calculates density aware chamfer distance loss. 
    Follows the "Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion" paper.
    This is implemented only for the case when the number of points for pred_pcl and target_pcl are the same.

    Parameters
    ----------
    pred_pcl: torch.Tensor
        Predicted point cloud output from a deep learning model. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    target_pcl: torch.Tensor
        Target point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    alpha: float
        Hyperparameter in front of the L2 distance as mentioned in the paper. 
        Default is 50.0
        
    Returns
    -------
    torch.Tensor
        In (1, 1) format. It's the loss value.
    """
    s1 = pred_pcl.detach().clone()
    s2 = target_pcl.detach().clone()
    
    device, dtype = s1.device, s1.dtype
    wrt_s1_argmin, wrt_s1_min_freq = _argmin_freq(s1, s2, device=device, dtype=dtype)
    wrt_s2_argmin, wrt_s2_min_freq = _argmin_freq(s2, s1, device=device, dtype=dtype)

    L1 = _one_way_chamfer_dist(pred_pcl, target_pcl, wrt_s1_argmin, wrt_s1_min_freq, alpha, device, dtype)
    L2 = _one_way_chamfer_dist(target_pcl, pred_pcl, wrt_s2_argmin, wrt_s2_min_freq, alpha, device, dtype)

    loss = 0.5 * (L1 + L2)

    return loss

def pcl_chamfer_dist(pred_pcl: torch.Tensor, 
                     target_pcl: torch.Tensor, 
                     reg: float, 
                     version: str='dcd', 
                     alpha: float=50.0) -> torch.Tensor:
    """
    Description
    -----------
    Calculates Chamfer distance between 2 point clouds. Either a vanilla Chamfer distance or 
    density-aware Chamfer distance presented in the "Density-aware Chamfer Distance 
    as a Comprehensive Metric for Point Cloud Completion" paper.

    Parameters
    ----------
    pred_pcl: torch.Tensor
        Predicted point cloud output from a deep learning model. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    target_pcl: torch.Tensor
        Target point cloud. 
        In (B, N, 3) format, where B is the batch dimension and N is the number of points in the point cloud.
    reg: float
        Regularization term for the loss.
    version: str
        'dcd' for density-aware Chamfer distance and 'cd' for vanilla Chamfer distance.
    alpha: float
        Hyperparameter in front of the L2 distance as mentioned in the paper. 
        Default is 50.0
        
    Returns
    -------
    torch.Tensor
        In (1, 1) format. It's the loss value.
    """
    loss = None
    if version == 'cd':
        loss = reg * chamfer_distance(pred_pcl, target_pcl)
    else:
        loss = reg * density_aware_chamfer_dist(pred_pcl, target_pcl, alpha)
        
    return loss

def cross_entropy_loss(pred: torch.Tensor, 
                       target: torch.Tensor, 
                       reg: float,
                       label_smoothing: float=0.0025) -> torch.Tensor:
    """
    Description
    -----------
    Calculates cross entropy loss, as presented in pytorch.
    Simply a wrapper to synchronize data type and device between pred and target.

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
    target = target.to(device=pred.device, dtype=pred.dtype)
    cross_ent_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss = cross_ent_loss(input=pred, target=target,)
    return reg * loss
