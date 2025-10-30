import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.utils.tensors import trunc_normal_


class VisionTransformerFramePredictor(nn.Module):
    """Frame-only Vision Transformer Predictor without action conditioning"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.use_activation_checkpointing = use_activation_checkpointing

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Create causal attention mask if needed
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            # Simple causal mask without action tokens
            attn_mask = torch.triu(torch.ones(grid_depth, grid_depth), diagonal=1).bool()
            attn_mask = attn_mask.repeat_interleave(grid_height * grid_width, dim=0)
            attn_mask = attn_mask.repeat_interleave(grid_height * grid_width, dim=1)
            self.register_buffer('attn_mask', attn_mask)
        else:
            self.attn_mask = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x):
        """
        Args:
            x: context frame tokens [B, T*H*W, D]
        Returns:
            Predicted next frame tokens [B, H*W, D]
        """
        # Map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N, D = x.size()
        
        # Get frame-level dimensions
        tokens_per_frame = self.grid_height * self.grid_width
        T = N // tokens_per_frame

        # Reshape to expose frame structure
        x = x.view(B, T, tokens_per_frame, D)  # [B, T, H*W, D]
        x = x.flatten(1, 2)  # [B, T*H*W, D]

        # Apply causal attention through blocks
        attn_mask = self.attn_mask[:x.size(1), :x.size(1)].to(x.device) if self.attn_mask is not None else None
        
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                )

        # Extract and reshape the last frame's worth of tokens
        x = x.view(B, T, tokens_per_frame, D)  # [B, T, H*W, D]
        x = x[:, -1]  # Take last frame's tokens [B, H*W, D]

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_frame_predictor(**kwargs):
    model = VisionTransformerFramePredictor(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model