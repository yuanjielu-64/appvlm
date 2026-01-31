"""
Qwen2.5-VL + DPT Head for Navigation Parameter Regression

参考 DUSt3R 的 DPT head 设计：
- 从 Qwen 的多层提取 hidden states
- 使用渐进式特征融合 (progressive refinement)
- 预测 7 个导航参数 (MSE regression)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class RegressionOutput(ModelOutput):
    """回归任务输出"""
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class HistoryEncoder(nn.Module):
    """
    轻量级历史帧编码器

    用 CNN 提取每帧特征，然后用 Transformer 聚合时序信息。
    只处理历史帧，当前帧由 VLM 处理。

    Args:
        img_size: 输入图像大小
        hidden_dim: 输出特征维度
        num_frames: 历史帧数量
        num_transformer_layers: Transformer 层数
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_frames: int = 2,
        num_transformer_layers: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # 轻量 CNN 提取每帧特征
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim)
        )

        # Transformer 聚合多帧时序信息
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, history_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_images: [B, num_frames, 3, H, W] 历史帧图像

        Returns:
            history_feat: [B, hidden_dim] 聚合后的历史特征
        """
        B, T, C, H, W = history_images.shape

        # 每帧过 CNN: [B*T, 3, H, W] -> [B*T, hidden_dim]
        imgs = history_images.view(B * T, C, H, W)
        feats = self.img_encoder(imgs)  # [B*T, hidden_dim]
        feats = feats.view(B, T, -1)    # [B, T, hidden_dim]

        # 加位置编码
        feats = feats + self.pos_embedding[:, :T, :]

        # Transformer 聚合时序
        feats = self.temporal_transformer(feats)  # [B, T, hidden_dim]

        # 取最后一帧的特征（最接近当前帧）
        history_feat = feats[:, -1, :]  # [B, hidden_dim]

        return history_feat


class FeatureRefinementBlock(nn.Module):
    """
    特征细化模块 - 参考 DUSt3R 的 RefineNet

    核心设计:
    1. 残差连接保留信息
    2. 多层卷积进行特征细化
    3. 支持跳跃连接融合不同层级特征

    Args:
        features: 特征维度
        use_bn: 是否使用 BatchNorm (默认不用,与 DUSt3R 一致)
    """
    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()

        # 跳跃连接投影 (参考 DUSt3R 的 feature alignment)
        self.resample = nn.Conv1d(features, features, kernel_size=1, bias=False)

        # 双层卷积 refinement (参考 DUSt3R 的 residual refinement)
        self.conv1 = nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False if use_bn else True)
        self.conv2 = nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False if use_bn else True)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(features)
            self.bn2 = nn.BatchNorm1d(features)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # 初始化 (参考 DUSt3R)
        self._init_weights()

    def _init_weights(self):
        """权重初始化 - 参考 DUSt3R"""
        for m in [self.resample, self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, C, L] 当前层特征 (来自高层)
            skip: [B, C, L] 跳跃连接 (来自低层,可选)
        Returns:
            refined: [B, C, L] 融合并细化后的特征
        """
        residual = x

        # 1. 跳跃连接融合 (如果提供)
        if skip is not None:
            x = x + self.resample(skip)

        # 2. 第一层卷积 + 激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3. 第二层卷积 + 激活
        out = self.conv2(out)
        out = self.bn2(out)

        # 4. 残差连接 (参考 DUSt3R 的 residual design)
        out = out + residual
        out = self.relu(out)

        return out


class SimpleMLP(nn.Module):
    """
    方案 A：Simple MLP Head

    只使用 last_hidden_state 的最后一个 token
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_params: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_params = num_params

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_params)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_state: [B, seq_len, hidden_size]
        Returns:
            predictions: [B, num_params]
        """
        # 只取最后一个 token
        last_token_feature = last_hidden_state[:, -1, :]  # [B, hidden_size]
        predictions = self.mlp(last_token_feature)
        return predictions


class TransformerHead(nn.Module):
    """
    方案 B：Transformer Head

    在 last_hidden_state 上再做一次 self-attention，
    然后全局池化所有 token
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_params: int = 7,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_params = num_params

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

        # Regression MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_params)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_state: [B, seq_len, hidden_size]
        Returns:
            predictions: [B, num_params]
        """
        # Self-attention (学习 token 之间的关系)
        attn_out, _ = self.attention(
            last_hidden_state,
            last_hidden_state,
            last_hidden_state
        )
        x = self.norm1(last_hidden_state + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # [B, seq_len, hidden_size]

        # Global average pooling (所有 token)
        pooled = x.mean(dim=1)  # [B, hidden_size]

        # Regression
        predictions = self.mlp(pooled)
        return predictions


class DPTHead(nn.Module):
    """
    DPT Head - 参考 DUSt3R 的多层特征融合设计

    核心设计 (基于 DUSt3R):
    1. **多层特征提取**: 从 VLM 不同深度提取 hidden states
    2. **投影到统一空间**: 将不同层投影到相同特征维度
    3. **渐进式融合**: 从高层到低层逐步融合 (top-down refinement)
    4. **残差细化**: 使用 RefineNet blocks 进行特征细化
    5. **注意力池化**: 学习序列中哪些位置对回归重要
    6. **回归预测**: 最终 MLP 输出参数

    与 DUSt3R 的区别:
    - DUSt3R: 2D 空间操作 (Conv2d), 输出密集预测 (per-pixel depth)
    - 本实现: 1D 序列操作 (Conv1d), 输出全局参数 (regression)

    Args:
        hidden_size: VLM hidden state 维度 (1536 for Qwen2.5-VL-3B)
        num_params: 回归参数数量 (7 for navigation)
        feature_dim: 中间特征维度 (256, 参考 DUSt3R)
        num_layers: 提取的层数 (4, 参考 DUSt3R 的 multi-scale)
        dropout: Dropout 率
    """

    def __init__(
        self,
        hidden_size: int = 1536,
        num_params: int = 7,
        feature_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_history: bool = False,
        history_dim: int = 256,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_params = num_params
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.use_history = use_history
        self.history_dim = history_dim

        # 1. 每层的投影 (参考 DUSt3R 的 ViT feature projection)
        # hidden_size → feature_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, feature_dim),
                nn.LayerNorm(feature_dim)  # 添加 LayerNorm 稳定训练
            )
            for _ in range(num_layers)
        ])

        # 2. 特征融合模块 (参考 DUSt3R 的 progressive refinement)
        # 从高层到低层: layer[3] → layer[2] → layer[1] → layer[0]
        self.fusion_blocks = nn.ModuleList([
            FeatureRefinementBlock(feature_dim, use_bn=False)
            for _ in range(num_layers - 1)
        ])

        # 3. Spatial attention pooling (学习序列中哪些 token 重要)
        # 类似 DUSt3R 中的 spatial aggregation
        self.spatial_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Softmax(dim=1)
        )

        # 4. 历史特征融合层 (可选)
        if use_history:
            # 将 VLM 特征和历史特征融合
            self.history_fusion = nn.Sequential(
                nn.Linear(feature_dim + history_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # 5. Regression MLP (全局特征 → 参数)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_params)
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化 - 参考 DUSt3R"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        multi_layer_hidden_states: List[torch.Tensor],
        history_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 - 参考 DUSt3R 的 multi-scale fusion

        Args:
            multi_layer_hidden_states: List of [B, seq_len, hidden_size]
                                       长度为 num_layers (通常是 4)
                                       顺序: [layer_-4, layer_-3, layer_-2, layer_-1]
                                       (从低层到高层)
            history_feat: [B, history_dim] 历史帧特征 (可选)

        Returns:
            predictions: [B, num_params] 回归参数预测

        处理流程:
            1. 投影: 每层 [B, seq_len, hidden_size] → [B, seq_len, feature_dim]
            2. 融合: 高层到低层渐进式融合 (top-down)
            3. 池化: 注意力加权池化 → [B, feature_dim]
            4. 历史融合: 与历史特征融合 (可选)
            5. 回归: MLP → [B, num_params]
        """
        B, seq_len, _ = multi_layer_hidden_states[0].shape

        # Step 1: 投影所有层到统一特征空间
        # 参考 DUSt3R 的 feature projection
        projected = [
            proj(hidden_state)  # [B, seq_len, feature_dim]
            for proj, hidden_state in zip(self.projections, multi_layer_hidden_states)
        ]

        # Step 2: 渐进式融合 (参考 DUSt3R 的 top-down refinement)
        # 转换为 [B, feature_dim, seq_len] 以使用 Conv1d
        projected = [p.transpose(1, 2) for p in projected]  # [B, feature_dim, seq_len]

        # 从最高层开始 (projected[-1] = layer_-1, 最深层)
        fused = projected[-1]

        # 渐进式向下融合: layer[-1] → layer[-2] → ... → layer[0]
        # 类似 DUSt3R 的 decoder path
        for i in range(len(self.fusion_blocks) - 1, -1, -1):
            skip = projected[i]  # 来自更浅层的特征
            fused = self.fusion_blocks[i](fused, skip)  # 融合并细化

        # 转回 [B, seq_len, feature_dim]
        fused = fused.transpose(1, 2)

        # Step 3: Spatial attention pooling
        # 学习哪些 token 位置对回归任务重要
        # (图像 token vs 文本 token, 不同空间位置的重要性)
        attention_weights = self.spatial_attention(fused)  # [B, seq_len, 1]
        pooled = (fused * attention_weights).sum(dim=1)  # [B, feature_dim]

        # Step 4: 历史特征融合 (可选)
        if self.use_history and history_feat is not None:
            # concat VLM 特征和历史特征，然后融合
            combined = torch.cat([pooled, history_feat], dim=-1)  # [B, feature_dim + history_dim]
            pooled = self.history_fusion(combined)  # [B, feature_dim]

        # Step 5: Regression MLP
        predictions = self.mlp(pooled)  # [B, num_params]

        return predictions


# 为了向后兼容，保留 RegressionHead 别名
RegressionHead = SimpleMLP


class Qwen2_5_VLForRegression(nn.Module):
    """
    Qwen2.5-VL + Regression Head for 导航参数回归

    支持三种 head 类型：
    - 'simple_mlp': 只用最后一个 token（最快，参数最少）
    - 'transformer': 在 last_hidden_state 上做 attention（中等）
    - 'dpt': 多层特征融合 + 空间 attention（最强，最慢）

    可选支持历史帧编码 (use_history=True)：
    - 历史帧通过轻量 CNN + Transformer 编码
    - 与 VLM 特征融合后送入回归头
    """

    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        num_params: int = 7,
        head_type: str = 'simple_mlp',  # 'simple_mlp', 'transformer', 'dpt'
        num_layers_to_extract: int = 4,  # DPT 用
        dropout: float = 0.1,
        use_history: bool = False,  # 是否使用历史帧
        num_history_frames: int = 2,  # 历史帧数量
        history_dim: int = 256,  # 历史特征维度
        param_weights: Optional[List[float]] = None,  # 参数权重
    ):
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config
        self.head_type = head_type
        self.num_layers_to_extract = num_layers_to_extract
        self.use_history = use_history
        self.num_history_frames = num_history_frames
        self.history_dim = history_dim

        # 注册参数权重为 buffer (不参与梯度更新)
        if param_weights is not None:
            self.register_buffer(
                'param_weights',
                torch.tensor(param_weights, dtype=torch.float32)
            )
        else:
            self.param_weights = None

        # 历史帧编码器 (可选)
        if use_history:
            self.history_encoder = HistoryEncoder(
                hidden_dim=history_dim,
                num_frames=num_history_frames,
                num_transformer_layers=2,
            )

        # 根据 head_type 创建不同的回归头
        if head_type == 'simple_mlp':
            self.regression_head = SimpleMLP(
                hidden_size=self.config.hidden_size,
                num_params=num_params,
                dropout=dropout,
            )
        elif head_type == 'transformer':
            self.regression_head = TransformerHead(
                hidden_size=self.config.hidden_size,
                num_params=num_params,
                num_attention_heads=8,
                dropout=dropout,
            )
        elif head_type == 'dpt':
            self.regression_head = DPTHead(
                hidden_size=self.config.hidden_size,
                num_params=num_params,
                feature_dim=256,
                num_layers=num_layers_to_extract,
                dropout=dropout,
                use_history=use_history,
                history_dim=history_dim,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Choose from ['simple_mlp', 'transformer', 'dpt']")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,  # [B, num_params]
        history_images: Optional[torch.Tensor] = None,  # [B, num_frames, 3, H, W]
        **kwargs
    ) -> RegressionOutput:
        """
        Args:
            labels: [B, num_params] 回归目标（7个导航参数）
            history_images: [B, num_frames, 3, H, W] 历史帧图像 (可选)

        Returns:
            RegressionOutput(loss, predictions, hidden_states)
        """
        # 1. 前向传播 base_model (处理当前帧)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        # 2. 编码历史帧 (可选)
        history_feat = None
        if self.use_history and history_images is not None:
            history_feat = self.history_encoder(history_images)  # [B, history_dim]
            # print(f"✓ history_images shape: {history_images.shape}")  # 应该是 [B, 2, 3, 224, 224]
            # print(f"✓ history_feat shape: {history_feat.shape}")

        # 3. 根据 head_type 准备输入并预测
        if self.head_type == 'dpt':
            # DPT 需要多层 hidden states
            all_hidden_states = outputs.hidden_states
            selected_layers = all_hidden_states[-self.num_layers_to_extract:]
            predictions = self.regression_head(selected_layers, history_feat=history_feat)
        else:
            # SimpleMLP 和 TransformerHead 都用最后一层
            last_hidden_state = outputs.hidden_states[-1]
            predictions = self.regression_head(last_hidden_state)

        # 4. 计算加权 MSE 损失
        loss = None
        if labels is not None:
            if self.param_weights is not None:
                # 加权 MSE: weighted_mse = mean(weight * (pred - label)^2)
                squared_errors = (predictions - labels) ** 2  # [B, num_params]
                weights = self.param_weights.to(squared_errors.device)  # [num_params]
                weighted_errors = squared_errors * weights  # [B, num_params]
                loss = weighted_errors.mean()
            else:
                # 普通 MSE
                loss_fct = nn.MSELoss()
                loss = loss_fct(predictions, labels)

        return RegressionOutput(
            loss=loss,
            predictions=predictions,
            hidden_states=None,  # 不再返回 hidden_states
        )

    @torch.no_grad()
    def predict(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        history_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理接口"""
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            history_images=history_images,
        )
        return output.predictions

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # 保存 base_model (包含 LoRA 权重)
        self.base_model.save_pretrained(os.path.join(save_directory, "base_model"))

        # 保存 regression_head (移到 CPU 以兼容 DeepSpeed)
        regression_head_state = {
            key: value.cpu() for key, value in self.regression_head.state_dict().items()
        }
        torch.save(
            regression_head_state,
            os.path.join(save_directory, "regression_head.pth")
        )

        # 保存 history_encoder (如果有)
        if self.use_history:
            history_encoder_state = {
                key: value.cpu() for key, value in self.history_encoder.state_dict().items()
            }
            torch.save(
                history_encoder_state,
                os.path.join(save_directory, "history_encoder.pth")
            )

        # 保存配置
        config = {
            "num_params": self.regression_head.num_params,
            "head_type": self.head_type,
            "num_layers_to_extract": self.num_layers_to_extract,
            "use_history": self.use_history,
            "num_history_frames": self.num_history_frames,
            "history_dim": self.history_dim,
        }
        with open(os.path.join(save_directory, "regression_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """加载预训练模型"""
        import os
        import json

        # 加载 base_model
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "base_model"),
            **kwargs
        )

        # 加载配置
        with open(os.path.join(pretrained_model_name_or_path, "regression_config.json")) as f:
            config = json.load(f)

        # 创建模型
        model = cls(
            base_model=base_model,
            num_params=config["num_params"],
            head_type=config.get("head_type", "simple_mlp"),  # 向后兼容
            num_layers_to_extract=config.get("num_layers_to_extract", 4),  # 向后兼容
            use_history=config.get("use_history", False),  # 向后兼容
            num_history_frames=config.get("num_history_frames", 2),  # 向后兼容
            history_dim=config.get("history_dim", 256),  # 向后兼容
        )

        # 加载 regression_head 权重
        regression_head_state = torch.load(
            os.path.join(pretrained_model_name_or_path, "regression_head.pth"),
            map_location="cpu"
        )
        model.regression_head.load_state_dict(regression_head_state)

        # 加载 history_encoder 权重 (如果有)
        history_encoder_path = os.path.join(pretrained_model_name_or_path, "history_encoder.pth")
        if model.use_history and os.path.exists(history_encoder_path):
            history_encoder_state = torch.load(history_encoder_path, map_location="cpu")
            model.history_encoder.load_state_dict(history_encoder_state)

        return model
