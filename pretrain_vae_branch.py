"""
from train.loss import VAE_Loss
import torch
from torch import nn

# Finalized hyperparameters...
k = 36
num_batches = 32
num_points = 9000
n_eig = 250
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
ntokens = int(1.5 * (num_points / k))
n_features=176
n_heads=4
drop_rate=0
latent_vec_len=3*n_eig
latent_vec_output_shape=(n_eig,3)
print(ntokens)
print(int(ntokens * (k / 1.5)))
print("Initializing model...")


model = VAE_Branch(input_dimension=k * k,
                    backbone_transformer_sequence_len=30,
                    vae_transformer_sequence_len=2,
                    n_features=n_features,
                    n_heads=n_heads,
                    dropout_rate=drop_rate,
                    n_tokens=ntokens,
                    latent_vec_len=latent_vec_len,
                    latent_vec_output_shape=latent_vec_output_shape,
                    device=device, 
                    dtype=dtype)

print("Initializing loss...")
loss = VAE_Loss(prior_mean=0.0,
                prior_var=1.0,
                total_data_num = k,
                batch_size=num_batches,
                epochs=1,
                has_true_latent_vec=True)

print("moving parameters to gpu...")
for param in model.parameters():
    param.data.fill_(0.5)
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



z, mean, std = model(input1, input2)
print(z.shape)
loss_val = loss(2, 2, target_rep_tensor, target_recon, mean, std, z, target_z)
print(loss_val)
loss_val.backward() 
optimizer.step()

from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
input1 = torch.ones((32, 176, 375)).to(device=device, dtype=dtype)
input1.requires_grad = True
l = nn.Linear(375, 9000, device=device, dtype=dtype)
#print(l(input1))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
input1 = torch.randn((2, 1, 4))
sm = nn.Softmax(dim=2)
print(sm(input1))

#print(input1)
input1 = input1.to(device=device, dtype=dtype)
#print(input1)

from functools import reduce
output = torch.ones(size=(2, 2, 2))
print(output)
output = output.view(output.shape[0], reduce((lambda x, y: x * y), output.shape[1:]))
print(output.shape)
print(output)
#print(output[1,:,:].shape)
"""

if __name__ == "__main__":
    
    import torch
    from models.models import VAE_MAE_Seg_Model
    from train.loss import VAE_MAE_Seg_Loss
    # Finalized hyperparameters...
    k = 30
    num_batches = 128
    num_points = 6000
    n_eig = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    n_tokens_scale = 2
    n_tokens = int(n_tokens_scale * (num_points / k))
    n_points_per_token = k#int((k / n_tokens_scale))
    print(device)
    print(f"{n_tokens} tokens")
    print(f"{n_points_per_token} points per token")
    n_features=176
    n_heads=4
    dropout_rate=0.0
    latent_vec_len=3*n_eig
    latent_vec_output_shape=(n_eig,3)


    model = VAE_MAE_Seg_Model(input_dimension= k * k,
                            backbone_transformer_sequence_len=6,
                            vae_transformer_sequence_len=2,
                            mae_transformer_sequence_len=2,
                            decoder_transformer_sequence_len=4,
                            seg_transformer_sequence_len=2,
                            vae_n_features=n_features,
                            vae_n_heads=n_heads,
                            vae_dropout_rate=dropout_rate,
                            vae_n_tokens=n_tokens,
                            vae_latent_vec_len=latent_vec_len,
                            vae_latent_vec_output_shape=latent_vec_output_shape,
                            backbone_n_features=n_features,
                            backbone_n_heads=n_heads,
                            backbone_dropout_rate=dropout_rate,
                            mae_enc_n_features=n_features,
                            mae_enc_n_heads=n_heads,
                            mae_enc_dropout_rate=dropout_rate,
                            mae_dec_n_features=n_features,
                            mae_dec_n_heads=n_heads,
                            mae_dec_n_dropout_rate=dropout_rate,
                            mae_output_dimension=3,
                            mae_n_tokens=n_tokens,
                            seg_enc_n_features=n_features,
                            seg_enc_n_heads=n_heads,
                            seg_enc_dropout_rate=dropout_rate,
                            seg_n_tokens=n_tokens,
                            n_points_per_token=n_points_per_token,
                            device=device,
                            dtype=dtype
                            )

    loss = VAE_MAE_Seg_Loss(total_data_num = k,
                            batch_size=num_batches,
                            epochs=1,
                            mae_loss_version='dcd')

    unmasked_input = torch.randn((num_batches, n_tokens, k * k)).to(device=device, dtype=dtype)
    input_positions = torch.randn((num_batches, 2*n_tokens, k * k)).to(device=device, dtype=dtype)
    target_rep_tensor = torch.randn((num_batches, num_points, n_eig)).to(device=device, dtype=dtype)
    target_recon = torch.randn((num_batches, num_points, 3)).to(device=device, dtype=dtype)
    target_z = torch.randn((num_batches, n_eig, 3)).to(device=device, dtype=dtype)

    import random

    unmasked_token_indices = []
    masked_token_indices = []
    for i in range(num_batches):
        rnd = list(range(2 * n_tokens))
        random.shuffle(rnd)
        unmasked_token_indices.append(rnd[:n_tokens])
        masked_token_indices.append(rnd[n_tokens:])


    for param in model.parameters():
        param.data.fill_(0.2)
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    z, mean, std, mae_output, seg_output = model(unmasked_input,
                                                unmasked_token_indices,
                                                masked_token_indices,
                                                input_positions)
    print(z.shape)
    print(mae_output.shape)
    print(seg_output.shape)
    print("Calculating loss...")
    loss_val = loss(mae_pred=mae_output,
                    mae_target=torch.randn(*mae_output.shape).to(device=device, dtype=dtype),
                    seg_pred=seg_output,
                    seg_target=torch.randn(*seg_output.shape).to(device=device, dtype=dtype),
                    vae_cur_iter=2,
                    vae_cur_epoch=2,
                    vae_target_reprojection_tensor=target_rep_tensor,
                    vae_target_recon=target_recon,
                    vae_pred_mean=mean,
                    vae_pred_std=std,
                    vae_pred_z=z,
                    vae_target_z=target_z,
    )
    print("Calculating backward graph...")
    #print(loss_val)
    loss_val.backward() 
    print("optimization step...")
    optimizer.step()
    """
    from data_process.utils.data_process import *
    import os

    file = os.path.join(os.getcwd(), 'sample_data/7PQAZ8X1_upper.obj')
    print("Reading point cloud...")
    point_cloud = read_point_cloud(file)
    print("Patchifying point cloud...")
    i, l, j, k = patchify_point_cloud(point_cloud)
    print("Patchified point cloud...")
    print(point_cloud.shape)
    print(i.shape)
    print(l.shape)
    """