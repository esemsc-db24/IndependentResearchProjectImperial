## Inputting train model rollout
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
import numpy as np
import torch.nn as nn



def train_model_aae_rollout(model,train_loader,val_loader,optimizer,discriminator,disc_optimizer, device,epochs=3,patience=5,min_delta=1e-4,
    adv_weight=0.01, scheduled_sampling_start=1.0, scheduled_sampling_end=0.0, scheduled_sampling_decay=0.95,
    block_size=None, num_blocks=None
):
    """
    Train out Adversarial Autoencoder (AAE) with block-wise rollout and scheduled sampling.

    Steps:
      1. Train discriminator to distinguish real vs fake latent vectors (adversarial regularization).
      2. Train model to minimize reconstruction error across multiple forecast blocks,
         while fooling the discriminator with realistic latent codes.
      3. Use scheduled sampling: gradually replace teacher-forcing with model’s own predictions
         during rollout.
      4. Monitor validation loss and apply early stopping + learning rate scheduling.

    Args:
        model: the main forecasting model (autoencoder / VAE-style).
        train_loader: DataLoader for training batches.
        val_loader: DataLoader for validation batches.
        optimizer: optimizer for model parameters.
        discriminator: adversarial discriminator on latent z.
        disc_optimizer: optimizer for discriminator parameters.
        device: torch device ("cpu" or "cuda").
        epochs: number of epochs to train.
        patience: patience for early stopping.
        min_delta: minimum improvement in val loss to reset patience.
        adv_weight: weight of adversarial loss relative to reconstruction.
        scheduled_sampling_start: initial teacher-forcing probability (1.0 = always).
        scheduled_sampling_end: minimum probability at end of training.
        scheduled_sampling_decay: decay factor per epoch.
        block_size: number of timesteps per forecast block.
        num_blocks: number of blocks to rollout during training.

    Returns:
        model: the trained model.
    """
    # Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    ## the below is useful to mantain the best loss
    adversarial_loss = BCELoss()
    best_val_loss = float('inf')
    patience_cnt = 0

    # initialise teacher-forcing probability:
    sampling_prob = scheduled_sampling_start

    for epoch in range(1, epochs + 1):
        ## putting model and discriminator in training mode
        model.train()
        discriminator.train()
        total_train_loss = 0.0
    ## Here I am looping over training batches and extracting past context and targets
    ## Reshaping to be able to compare prediction with ground truth
        for inputs, full_targets in train_loader:
            inputs      = inputs.to(device)            # [B, win, F]
            full_targets= full_targets.to(device)      # [B, blk*#blks, F]
            B, _, F_in  = inputs.shape
            targets     = full_targets.view(B, num_blocks, block_size, F_in)

            # 1st Phase of training --> Trainining discriminator 
            ## Rearranging input shape for transformer
            inputs_enc = inputs.permute(1,0,2)         # [win, B, F]
            with torch.no_grad():
                # Below I am running the encoder, get latent distribution parameters (mu, logvar).
                # Then we sample from this latent distribution to create z_fake.
                # It’s fake because it’s model-generated latent codes, not guaranteed to follow Gaussian
                _, mu, logvar, _ = model(inputs_enc)
                z_fake = model.reparameterize(mu, logvar).detach()
            # Then we sample a real latent vector from a standard Gaussian prior (same shape as z_fake).
            # This is the ideal distribution we want the encoder to match. rand_like creates
            # random numbers drawn from a Gaussian 𝒩(0, 1).
            z_real      = torch.randn_like(z_fake)
            real_lbls   = torch.ones(z_real.size(0),1, device=device)
            fake_lbls   = torch.zeros(z_fake.size(0),1, device=device)

            # Then we pass z_real and z_fake through the discriminator
            # and we calculate disc_loss 
            disc_optimizer.zero_grad()
            d_real = discriminator(z_real) ## output a prob of close to 1 if it thinks it's real
            d_fake = discriminator(z_fake) ## output close to 0 if it thinks its fake
            # total loss encourage 1 for real latent and 0 for fake ones
            disc_loss = adversarial_loss(d_real, real_lbls) \
                      + adversarial_loss(d_fake, fake_lbls)
            disc_loss.backward()
            disc_optimizer.step()

            # 2nd Phase of training --> Block-wise rollout + Adversarial regularisation
            optimizer.zero_grad()
            # input sequence
            current_win    = inputs_enc.clone()       # [win, B, F]
            total_rec_loss = 0.0

            for b in range(num_blocks):
                # 1) Predicting next time block from current window
                frc_block, mu, logvar, _ = model(current_win)
                # ensure shape [B, blk, F]
                if frc_block.dim()==2:
                    frc_block = frc_block.unsqueeze(1)

                # 2) Getting the true block so that I can compare
                true_block = targets[:, b]           # [B, blk, F]
                if true_block.dim()==2:
                    true_block = true_block.unsqueeze(1)

                # 3) Computing reconstruction loss
                rec_loss = F.mse_loss(frc_block, true_block, reduction='mean')
                total_rec_loss += rec_loss

                # 4) Picking next input via scheduled sampling:
                # With probability of sampling_prob, feed the ground truth as next input.
                # Otherwise, feed the model’s own prediction.
                if torch.rand(1).item() < sampling_prob:
                    next_block = true_block
                else:
                    next_block = frc_block.detach()

                # 5) Sliding the window and going to the next block
                #    convert [B, blk, F] → [blk, B, F]
                next_block = next_block.permute(1,0,2)
                current_win = torch.cat([current_win[block_size:], next_block], dim=0)

            # 6) Adversarial loss on the final latent z
            # Forcing the encoder’s latent distribution to mimic the true Gaussian prior 
            # So the latent space is variational
            z_gen    = model.reparameterize(mu, logvar)
            pred_lbls= torch.ones(z_gen.size(0),1, device=device)
            adv_loss = adversarial_loss(discriminator(z_gen), pred_lbls)

            # 7) Total loss & backward
            # total_rec_loss → measures how well the model forecasts blocks.
            # adv_loss → measures how realistic the latent space looks to the discriminator.
            # adv_weight → controls how much adversarial regularisation matters relative to frc.
            loss = total_rec_loss + adv_weight * adv_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # Validation 
        model.eval(); discriminator.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, full_targets in val_loader:
                inputs, full_targets = inputs.to(device), full_targets.to(device)
                inp = inputs.permute(1,0,2)
                frc_blk, _, _, _ = model(inp)
                if frc_blk.dim()==2:
                    frc_blk = frc_blk.unsqueeze(1)

                true_blk = full_targets[:, :block_size, :]
                if true_blk.dim()==2:
                    true_blk = true_blk.unsqueeze(1)
        # Here we are computing the loss between what we want to predict and ground truth     
                total_val_loss += F.mse_loss(frc_blk, true_blk, reduction='mean').item()
        
        avg_val = total_val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train={avg_train:.6f} | Val={avg_val:.6f} | "
            f"Disc={disc_loss.item():.4f} | Adv={adv_loss.item():.4f} | "
            f"SampProb={sampling_prob:.3f}"
        )

        # If validation loss improves by more than min_delta, reset patience counter.
        # Otherwise, increase patience counter. If counter reaches patience, we will stop training early.
        if best_val_loss - avg_val > min_delta:
            best_val_loss = avg_val
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"→ Early stopping at epoch {epoch}")
                break

        # Gradually makes the model rely less on ground truth, more on its own predictions.
        sampling_prob = max(scheduled_sampling_end,
                            sampling_prob * scheduled_sampling_decay)

    return model


