import torch
from typing import Optional


def move_to_device(input1: torch.Tensor, 
                   input2: Optional[torch.Tensor]=None, 
                   device: Optional[torch.device]=None,
                   dtype: Optional[torch.dtype]=None) -> list[Optional[torch.Tensor]]:
    """
    Description
    -----------
    Takes in 2 torch tensors and assign them to appropriate device so that they're pinned in the same device.

    Parameters
    ----------
    input1: torch.Tensor
        Input torch tensor
    input2: input2: Optional[torch.Tensor]
        Optional input torch tensor
    device: Optional[torch.device]
        Device to which the model parameters are mapped (ex. cpu or gpu).
    dtype: Optional[torch.dtype]
        Data type of the model parameters.
    
    Returns
    -------
    list[Optional[torch.Tensor]]
        Returns a list of torch tensors, in which the second output tensor may be a None.
    """
    if device is not None and input1.device != device:
        input1 = input1.to(device=device, dtype=dtype)
    if input2 is not None and (input1.device != input2.device or input1.dtype != input2.dtype):
        input2 = input2.to(device=input1.device, dtype=input1.dtype)
    return input1, input2

def batch_index_rearrange(input: torch.Tensor, batch_indices: list[list[int]]) -> torch.Tensor:
    """
    Description
    -----------
    Takes in a torch.Tensor in (num_batch, num_tokens, num_features) format 
    and a list of lists such that there are num_batch lists and each list has num_tokens size.
    batch_indices[batch][from] gives an index to which torch.Tensor(batch, from, :) should move to. 
    ie. torch.Tensor(batch, batch_indices[batch][from], :) = torch.Tensor(batch, from, :)

    Parameters
    ----------
    input: torch.Tensor
        Tensor in (num_batch, num_tokens, num_features) format.
    batch_indices: list[list[int]]
        list of index for each batch. len(batch_indices) == num_batch
        Each list of index is ordered in a new rearranged format that the input will be in.
    
    Returns
    -------
    torch.Tensor
        Returns a rearranged Tensor in the same dimension as the input.
    """
    num_batch, _, _ = input.shape
    rearranged_tensor = torch.empty(size=input.shape, dtype=input.dtype, device=input.device)
    for batch in range(num_batch):
        for src_i, dst_i in enumerate(batch_indices[batch]):
            rearranged_tensor[batch, dst_i, :] = input[batch, src_i, :]
    return rearranged_tensor

def batch_index_select(input: torch.Tensor, batch_indices: list[list[int]]) -> torch.Tensor:
    """
    Description
    -----------
    Takes in a torch.Tensor in (num_batch, num_tokens, num_features) format 
    and a list of lists such that there are num_batch lists and each list has indices to select.
    
    ie. new_tensor(batch, :, :) = input(batch, batch_indices[batch], :)

    Parameters
    ----------
    input: torch.Tensor
        Tensor in (num_batch, num_tokens, num_features) format.
    batch_indices: list[list[int]]
        List of index list to select for each batch. len(batch_indices) == num_batch
    
    Returns
    -------
    torch.Tensor
        Returns a selected Tensor in (num_batch, len(batch_indices[0]), num_features) format.
    """
    
    batch_indices = torch.Tensor(batch_indices).type(torch.LongTensor).to(device=input.device)
    num_batch, _, num_features = input.shape
    selected_num_tokens = len(batch_indices[0])
    selected_tensor = torch.empty(size=(num_batch, selected_num_tokens, num_features), 
                                  dtype=input.dtype, 
                                  device=input.device)

    for batch, index in enumerate(batch_indices):
        selected_tensor[batch, :, :] = torch.index_select(input=input[batch, :, :], 
                                                          dim=0, 
                                                          index=index)
    
    return selected_tensor
