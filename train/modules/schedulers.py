import math
from typing import Optional, Callable


def iter_per_epoch(total_data_num: int, batch_size: int) -> int:
    """
    Description
    -----------
        Returns the Number of iterations per epoch.

    Parameters
    ----------
    total_data_num: int
        Total number of data.
    batch_size: int
        Batch size for batch training.

    Returns
    -------
    int 
        Returns the Number of iterations per epoch.
    """
    return int(math.ceil(float(total_data_num) / float(batch_size)))

def cur_iteration(cur_iter: int, iter_per_epoch: int, cur_epoch: int) -> int:
    """
    Description
    -----------
        Returns the current iteration number within the scope of the total number of iterations
        that the model goes through. 

    Parameters
    ----------
    cur_iter: int
        Current iteration number within the scope of the current epoch
    iter_per_epoch: int
        Number of iterations per epoch.
    cur_epoch: int  
        Current training epoch that the model is going through.

    Returns
    -------
    int 
        Returns the current iteration number within the scope of the total number of iterations
        that the model goes through. 
    """
    return cur_iter + iter_per_epoch * cur_epoch

def total_training_iterations(total_data_num: int, batch_size: int, epochs: int) -> int:
    """
    Description
    -----------
    Total number of training iterations that the deep learning model will go through.
    
    Parameters
    ----------
    total_data_num: int
        Total number of data.
    batch_size: int
        Batch size for batch training.
    epochs: int
        Number of epochs that the deep learning model will be trained for.

    Returns
    -------
    int
        Returns the total number of training iterations.
    """
    return int(math.ceil(float(total_data_num) / float(batch_size)) * float(epochs))

def cyclic_annealing_scheduler(iteration: int, 
                               total_iterations: int, 
                               mono_inc: Optional[Callable[[float], float]]=None, 
                               num_cycles: int=4, 
                               beta_scale: float = 0.5,) -> float:
    """
    Description
    -----------
    This follows "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing".
    This is for training VAE.
    
    Parameters
    ----------
    iterations: int
        Current training iteration number within the scope of the entire training iterations.
    total_iterations: int
        Total number of training iterations that the deep learning model will go through.
    mono_inc: function
        Monotonically increasing function that takes in a float.\n
        mono_inc(0) = 0 and mono_inc(beta_scale) = 1
    num_cycles: int
        Number of cycles that the training process is divided into. (Default 4)
    beta_scale: float
        Proportion used to increase beta within a cycle (Default 0.5)
    
    Returns
    -------
    float
        Returns regularization factor beta for KL divergence loss.
    """
    
    T_M = float(total_iterations) / float(num_cycles)
    threshold = float((iteration - 1) % int(math.ceil(T_M))) / T_M

    if threshold <= beta_scale:
        if mono_inc is None:
            mono_inc = lambda x : x / beta_scale
        return mono_inc(threshold)
    else:
        return 1.0
