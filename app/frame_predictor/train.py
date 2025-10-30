import os
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa.transforms import make_transforms
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.datasets.data_manager import init_data
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

# -- Constants
log_freq = 10
CHECKPOINT_FREQ = 1
_GLOBAL_SEED = 0

logger = get_logger(__name__)

def init_frame_predictor(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_activation_checkpointing=False,
):
    """Initialize the frame predictor model."""
    from src.models import vision_transformer as video_vit
    from src.models.frame_predictor import vit_frame_predictor

    # Initialize the encoder (for context frames)
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    # Initialize the predictor
    predictor = vit_frame_predictor(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    encoder.to(device)
    predictor.to(device)

    return encoder, predictor

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
):
    """Initialize optimizer and schedulers."""
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler

def main(args):
    """Main training loop."""
    # -- Initialize random seeds
    random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    # -- Initialize distributed training
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- Set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device)

    # -- Get config parameters
    cfgs_meta = args.get("meta", {})
    cfgs_model = args.get("model", {})
    cfgs_data = args.get("data", {})
    cfgs_opt = args.get("optimization", {})
    cfgs_loss = args.get("loss", {})

    # -- Setup logging
    folder = args.get("folder", ".")
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        mode="+a",
    )

    # -- Initialize models
    encoder, predictor = init_frame_predictor(
        device=device,
        patch_size=cfgs_data.get("patch_size", 16),
        max_num_frames=max(cfgs_data.get("dataset_fpcs", [16])),
        tubelet_size=cfgs_data.get("tubelet_size", 2),
        model_name=cfgs_model.get("model_name", "vit_base"),
        crop_size=cfgs_data.get("crop_size", 224),
        pred_depth=cfgs_model.get("pred_depth", 6),
        pred_num_heads=cfgs_model.get("pred_num_heads"),
        pred_embed_dim=cfgs_model.get("pred_embed_dim", 384),
        uniform_power=cfgs_model.get("uniform_power", False),
        use_sdpa=cfgs_meta.get("use_sdpa", False),
        use_rope=cfgs_model.get("use_rope", False),
        use_silu=cfgs_model.get("use_silu", False),
        use_activation_checkpointing=cfgs_model.get("use_activation_checkpointing", False),
    )

    # -- Initialize target encoder (teacher)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- Setup data loading
    transform = make_transforms(
        random_horizontal_flip=cfgs_data.get("horizontal_flip", False),
        auto_augment=cfgs_data.get("auto_augment", False),
        crop_size=cfgs_data.get("crop_size", 224),
    )

    # -- Initialize data loader
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data_path=cfgs_data.get("datasets", [""])[0],
        batch_size=cfgs_data.get("batch_size", 32),
        frames_per_clip=max(cfgs_data.get("dataset_fpcs", [16])),
        fps=cfgs_data.get("fps", 4),
        transform=transform,
        num_workers=cfgs_data.get("num_workers", 4),
        world_size=world_size,
        rank=rank,
    )

    # -- Initialize optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=cfgs_opt.get("ipe", len(unsupervised_loader)),
        start_lr=cfgs_opt.get("start_lr", 1e-4),
        ref_lr=cfgs_opt.get("lr", 1e-3),
        final_lr=cfgs_opt.get("final_lr", 0.0),
        warmup=cfgs_opt.get("warmup", 10),
        anneal=cfgs_opt.get("anneal", 0),
        num_epochs=cfgs_opt.get("epochs", 100),
        wd=cfgs_opt.get("weight_decay", 1e-6),
        final_wd=cfgs_opt.get("final_weight_decay", 1e-6),
        mixed_precision=cfgs_meta.get("dtype", "float32") != "float32",
    )

    # -- Wrap models with DDP
    encoder = DistributedDataParallel(encoder)
    predictor = DistributedDataParallel(predictor)
    target_encoder = DistributedDataParallel(target_encoder)

    # -- Training loop
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    momentum_scheduler = (1 - 0.9) * (1 - torch.arange(cfgs_opt.get("epochs", 100)) / cfgs_opt.get("epochs", 100)) + 0.9

    for epoch in range(cfgs_opt.get("epochs", 100)):
        logger.info(f"Epoch {epoch + 1}")
        
        for itr, sample in enumerate(unsupervised_loader):
            itr_start_time = time.time()

            # Load video clips
            clips = sample[0].to(device, non_blocking=True)  # [B, C, T, H, W]
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # Forward through target encoder
                with torch.no_grad():
                    h = target_encoder(clips)
                    if cfgs_loss.get("normalize_reps", True):
                        h = F.layer_norm(h, (h.size(-1),))

                # Forward through context encoder and predictor
                z = encoder(clips[:, :, :-1])  # Encode all but last frame
                if cfgs_loss.get("normalize_reps", True):
                    z = F.layer_norm(z, (z.size(-1),))
                
                # Predict next frame
                z_pred = predictor(z)

                # Compute loss on the last frame
                tokens_per_frame = z_pred.size(1)
                h_target = h[:, -tokens_per_frame:]  # Last frame's tokens
                loss = torch.mean(torch.abs(z_pred - h_target) ** cfgs_loss.get("loss_exp", 1.0))
                loss = loss / cfgs_loss.get("loss_exp", 1.0)

                # Backward and optimize
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

                # Update target encoder
                m = momentum_scheduler[epoch]
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

                return float(loss), _new_lr, _new_wd

            (loss, _new_lr, _new_wd), gpu_time = gpu_timer(train_step)
            iter_time = time.time() - itr_start_time

            # Update meters
            loss_meter.update(loss)
            iter_time_meter.update(iter_time * 1000.0)  # Convert to ms

            # Log progress
            if itr % log_freq == 0 or itr == len(unsupervised_loader) - 1:
                logger.info(
                    f"[{epoch + 1}, {itr}] "
                    f"loss: {loss_meter.avg:.3f} "
                    f"[lr: {_new_lr:.2e}] "
                    f"[wd: {_new_wd:.2e}] "
                    f"[iter time: {iter_time_meter.avg:.1f}ms]"
                )
                csv_logger.log(epoch + 1, itr, loss, iter_time * 1000.0, gpu_time, 0.0)

        # Save checkpoint
        if rank == 0 and (epoch % CHECKPOINT_FREQ == 0 or epoch == cfgs_opt.get("epochs", 100) - 1):
            save_dict = {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "epoch": epoch,
                "loss": loss_meter.avg,
            }
            torch.save(save_dict, os.path.join(folder, f"checkpoint_e{epoch}.pt"))