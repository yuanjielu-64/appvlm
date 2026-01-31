"""
Transformer-based Imitation Learning Model for Robot Navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class VisionEncoder(nn.Module):
    """使用预训练的 Vision Transformer 提取图像特征"""
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze=False):
        super().__init__()
        # 使用 timm 库加载预训练 ViT
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.vit.num_features  # 通常是 768 for ViT-Base

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 图像
        Returns:
            features: [B, feature_dim] 特征向量
        """
        return self.vit(x)


class TemporalTransformer(nn.Module):
    """时序 Transformer，融合历史帧信息"""
    def __init__(self, d_model=768, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 位置编码（可学习）
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, d_model))  # 最多支持10帧

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model] 时序特征序列
        Returns:
            output: [B, d_model] 融合后的特征
        """
        B, T, D = x.shape

        # 添加位置编码
        x = x + self.pos_embedding[:, :T, :]

        # Transformer 编码
        x = self.transformer(x)  # [B, T, d_model]

        # 取最后一帧的特征（代表当前时刻）
        output = x[:, -1, :]  # [B, d_model]

        return output


class NavigationTransformerIL(nn.Module):
    """
    完整的 Transformer Imitation Learning 模型

    输入:
        - current_image: 当前 costmap 图像
        - history_images: 历史帧图像 (默认2帧)
        - velocity_state: [linear_vel, angular_vel]

    输出:
        - parameters: 导航参数 (7-8维)

    支持的 Vision 模型:
        - vit_base_patch16_224: 86M params, feature_dim=768
        - vit_large_patch16_224: 304M params, feature_dim=1024
        - vit_huge_patch14_224: 632M params, feature_dim=1280
    """
    def __init__(
        self,
        num_params=8,               # 输出参数数量
        num_history_frames=2,       # 历史帧数量
        vision_model='vit_base_patch16_224',
        vision_pretrained=True,
        vision_freeze=False,
        d_model=None,               # Transformer 隐藏维度 (None=自动匹配vision)
        nhead=None,                 # 注意力头数 (None=自动)
        num_transformer_layers=4,
        dropout=0.1,
        velocity_dim=2,             # 速度状态维度
        use_velocity=True           # 是否使用速度信息
    ):
        super().__init__()

        self.num_history_frames = num_history_frames
        self.use_velocity = use_velocity

        # 1. Vision Encoder (共享权重)
        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            pretrained=vision_pretrained,
            freeze=vision_freeze
        )
        self.feature_dim = self.vision_encoder.feature_dim

        # 自动设置 d_model 和 nhead
        if d_model is None:
            d_model = self.feature_dim  # 768 for base, 1024 for large, 1280 for huge
        if nhead is None:
            # 自动计算合适的 head 数量
            nhead = d_model // 64  # 768->12, 1024->16, 1280->20

        self.d_model = d_model

        # 2. 特征投影层 (如果 vision feature_dim != d_model)
        if self.feature_dim != d_model:
            self.feature_proj = nn.Linear(self.feature_dim, d_model)
        else:
            self.feature_proj = None

        # 3. 时序 Transformer (融合当前帧 + 历史帧)
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dropout=dropout
        )

        # 4. 速度状态编码器 (可选) - 根据 d_model 调整大小
        vel_hidden = max(256, d_model // 4)
        if use_velocity:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(velocity_dim, vel_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(vel_hidden // 2, vel_hidden)
            )
            fusion_dim = d_model + vel_hidden
        else:
            fusion_dim = d_model

        # 5. 参数回归头 - 根据 fusion_dim 调整大小
        head_hidden1 = max(512, d_model)
        head_hidden2 = max(256, d_model // 2)
        self.param_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden1),
            nn.LayerNorm(head_hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden1, head_hidden2),
            nn.LayerNorm(head_hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden2, num_params)
        )

    def forward(self, current_image, history_images=None, velocity_state=None):
        """
        Args:
            current_image: [B, 3, H, W] 当前帧
            history_images: [B, num_history, 3, H, W] 历史帧 (可选)
            velocity_state: [B, 2] [linear_vel, angular_vel] (可选)

        Returns:
            params: [B, num_params] 预测的导航参数
        """
        B = current_image.shape[0]

        # 1. 提取当前帧特征
        current_feat = self.vision_encoder(current_image)  # [B, feature_dim]

        # 2. 提取历史帧特征
        if history_images is not None and history_images.shape[1] > 0:
            # history_images: [B, T, 3, H, W]
            T = history_images.shape[1]
            # Reshape: [B*T, 3, H, W]
            history_images_flat = history_images.reshape(B * T, 3,
                                                         history_images.shape[3],
                                                         history_images.shape[4])
            history_feats_flat = self.vision_encoder(history_images_flat)  # [B*T, feature_dim]
            # Reshape back: [B, T, feature_dim]
            history_feats = history_feats_flat.reshape(B, T, -1)

            # 拼接当前帧和历史帧: [B, T+1, feature_dim]
            all_feats = torch.cat([history_feats, current_feat.unsqueeze(1)], dim=1)
        else:
            # 没有历史帧，只用当前帧
            all_feats = current_feat.unsqueeze(1)  # [B, 1, feature_dim]

        # 3. 特征投影 (如果需要)
        if self.feature_proj is not None:
            all_feats = self.feature_proj(all_feats)  # [B, T+1, d_model]

        # 4. 时序 Transformer 融合
        fused_feat = self.temporal_transformer(all_feats)  # [B, d_model]

        # 5. 融合速度状态 (可选)
        if self.use_velocity and velocity_state is not None:
            vel_feat = self.velocity_encoder(velocity_state)  # [B, vel_hidden]
            fused_feat = torch.cat([fused_feat, vel_feat], dim=-1)  # [B, d_model + vel_hidden]

        # 6. 预测参数
        params = self.param_head(fused_feat)  # [B, num_params]

        return params

    def get_num_params(self):
        """返回模型总参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 简化版本：不使用历史帧
class SimpleTransformerIL(nn.Module):
    """简化版本，只使用当前帧 + 速度"""
    def __init__(
        self,
        num_params=8,
        vision_model='vit_base_patch16_224',
        vision_pretrained=True,
        dropout=0.1
    ):
        super().__init__()

        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            pretrained=vision_pretrained,
            freeze=False
        )

        # 速度编码
        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # 回归头
        self.param_head = nn.Sequential(
            nn.Linear(self.vision_encoder.feature_dim + 256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_params)
        )

    def forward(self, image, velocity_state):
        vision_feat = self.vision_encoder(image)
        vel_feat = self.velocity_encoder(velocity_state)
        fused = torch.cat([vision_feat, vel_feat], dim=-1)
        return self.param_head(fused)


if __name__ == "__main__":
    # 测试模型
    model = NavigationTransformerIL(
        num_params=8,
        num_history_frames=2,
        use_velocity=True
    )

    print(f"Total parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Trainable parameters: {model.get_trainable_params() / 1e6:.2f}M")

    # 测试前向传播
    batch_size = 4
    current_img = torch.randn(batch_size, 3, 224, 224)
    history_imgs = torch.randn(batch_size, 2, 3, 224, 224)
    velocity = torch.randn(batch_size, 2)

    output = model(current_img, history_imgs, velocity)
    print(f"Output shape: {output.shape}")  # [4, 8]
