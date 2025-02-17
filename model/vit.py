import torch
from torch import nn
import timm


def create_vit_model(pretrained, num_classes=1000):
    """
    Creates a ViT model for ImageNet classification.

    Args:
        pretrained (bool): If True, loads pre-trained weights. Defaults to False.
        num_classes (int, optional): Number of output classes (defaults to 1000 for ImageNet). Defaults to 1000.

    Returns:
        nn.Module: The created ViT model.
    """

    if pretrained:
        # Fine-tune a pre-trained model (freeze early layers if desired)
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # Optionally freeze early layers

        # Modify the final classification head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    else:
        # Create a ViT model with randomly initialized weights
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        # Modify the final classification head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        
    return model

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

# def get_sincos_encoding_from_tree(coordinates, embedding_dim):
#     freq = 1 / np.power(10000, (2 * (np.arange(embedding_dim) // 2)) / embedding_dim)
#     encodings = []
#     for x, y in coordinates:
#         x_enc = np.sin(x * freq[::2])  # Sine for x-coordinates
#         y_enc = np.cos(y * freq[1::2])  # Cosine for y-coordinates
#         encodings.append(np.concatenate([x_enc, y_enc]))
#     return np.array(encodings)

def get_sincos_encoding_from_tree(coordinates: torch.Tensor, embedding_dim: int):
    """
    Compute the sin-cos positional encodings for a batch of coordinates.

    Args:
        coordinates (torch.Tensor): Tensor of shape [B, L, 2] where B is the batch size,
                                     L is the sequence length, and 2 corresponds to (x, y) coordinates.
        embedding_dim (int): Dimension of the positional encoding.

    Returns:
        torch.Tensor: Sin-cos encoding of shape [B, L, embedding_dim].
    """
    B, L, _ = coordinates.shape

    # Compute frequency scaling factors [embedding_dim // 2]
    freq = 1.0 / torch.pow(10000, (2 * torch.arange(embedding_dim // 2)) / embedding_dim).to(coordinates.device)

    # Split coordinates into x and y components
    x_coords = coordinates[:, :, 0].unsqueeze(-1)  # Shape: [B, L, 1]
    y_coords = coordinates[:, :, 1].unsqueeze(-1)  # Shape: [B, L, 1]

    # Compute sine for x-coordinates and cosine for y-coordinates
    x_enc = torch.sin(x_coords * freq)  # Shape: [B, L, embedding_dim // 2]
    y_enc = torch.cos(y_coords * freq)  # Shape: [B, L, embedding_dim // 2]

    # Concatenate encodings along the last dimension
    encoding = torch.cat([x_enc, y_enc], dim=-1)  # Shape: [B, L, embedding_dim]
    
    return encoding
class PatchEmbedding2D(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, seq_length=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # self.projection = nn.Conv2d(
        #     in_channels,
        #     embed_dim,
        #     kernel_size=patch_size,
        #     stride=patch_size
        # )
        if seq_length==None:
            seq_length = (img_size // patch_size) ** 2
        else:
            seq_length = seq_length
        
        self.projection = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(
        #     torch.randn(1, seq_length + 1, embed_dim)
        # )
        
    def forward(self, x, coordinates=None):
        if coordinates!=None:
            pos_embed = get_sincos_encoding_from_tree(coordinates=coordinates, embedding_dim=self.embed_dim)
            # Append positional embedding for the CLS token as a zero vector or learnable parameter
            cls_pos_embed = torch.zeros(1, 1, self.embed_dim, device=self.cls_token.device)  
            pos_embed = torch.cat([cls_pos_embed.expand(pos_embed.size(0), -1, -1), pos_embed], dim=1)

        self.pos_embed = pos_embed
        # Convert image to patches
        B = x.size(0)
        x = self.projection(x)  # Shape: [B, embed_dim, H', W']
        # x = rearrange(x, 'b c h w -> b (h w) c')  # Shape: [B, N, embed_dim]
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, seq_length=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        if seq_length==None:
            seq_length = (img_size // patch_size) ** 2
        else:
            seq_length = seq_length
        
        self.projection = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_length + 1, embed_dim)
        )
        
    def forward(self, x):
        # Convert image to patches
        B = x.size(0)
        x = self.projection(x)  # Shape: [B, embed_dim, H', W']
        # x = rearrange(x, 'b c h w -> b (h w) c')  # Shape: [B, N, embed_dim]
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            qkv
        )
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        seq_length=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, seq_length=seq_length)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # Use the [CLS] token for classification
        x = self.head(x)
        return x
    
class VisionTransformer2DPos(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        seq_length=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, seq_length=seq_length)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, coordinates=None):
        # import pdb
        # pdb.set_trace()
        x = self.patch_embed(x, coordinates=coordinates)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # Use the [CLS] token for classification
        x = self.head(x)
        return x

class PatchSizeEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, seq_length):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        self.projection = nn.Linear(
            patch_size*patch_size,
            embed_dim,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Create the embedding layer
        self.patch_embed = nn.Parameter(
            torch.randn(1, self.seq_length + 1, self.embed_dim)
        )
        
    def forward(self, x):
         
        # Convert image to patches
        B = x.size(0)
        x = self.projection(x)  # Shape: [B, embed_dim, H', W']
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, N+1, embed_dim]
        
        # Add positional encoding
        x = x + self.patch_embed
        
        return x
    
class AF_ViT(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        seq_length=514,
    ):
        super().__init__()
        self.patch_embed = PatchSizeEmbedding(patch_size=8, embed_dim=embed_dim, seq_length=seq_length)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # Use the [CLS] token for classification
        x = self.head(x)
        return x