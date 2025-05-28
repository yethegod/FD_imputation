import torch
import numpy as np
from typing import Optional, Tuple
from torch import nn
from tqdm import tqdm

from fdiff.utils.fourier import dft, idft
from fdiff.models.score_models import ScoreModule
from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch


class FrequencyDomainImputer:
    """
    基于频域扩散的缺失值补全器，灵感来自DiffPure的对抗净化思想
    
    核心思想：
    1. 将缺失值填充为0
    2. 转换到频域
    3. 使用扩散模型进行"净化"来恢复缺失的信息
    4. 转换回时域
    """
    
    def __init__(
        self,
        score_model: ScoreModule,
        diffusion_steps: int = 100,
        noise_level: float = 0.1,
        preserve_observed: bool = True,
        max_iterations: int = 1,
    ):
        """
        Args:
            score_model: 预训练的分数模型
            diffusion_steps: 扩散步数
            noise_level: 添加的噪声水平 (对应DiffPure中的t*)
            preserve_observed: 是否在补全过程中保持观测值不变
            max_iterations: 最大迭代次数
        """
        self.score_model = score_model
        self.noise_scheduler = score_model.noise_scheduler
        self.diffusion_steps = diffusion_steps
        self.noise_level = noise_level
        self.preserve_observed = preserve_observed
        self.max_iterations = max_iterations
        
        # 设置扩散时间步
        self.noise_scheduler.set_timesteps(diffusion_steps)
        
    def create_missing_mask(
        self, 
        X: torch.Tensor, 
        missing_rate: float = 0.2, 
        pattern: str = "random"
    ) -> torch.Tensor:
        """
        创建缺失值掩码
        
        Args:
            X: 输入时间序列 (batch_size, max_len, n_channels)
            missing_rate: 缺失率
            pattern: 缺失模式 ("random", "block", "channel")
            
        Returns:
            mask: 缺失值掩码，1表示观测值，0表示缺失值
        """
        batch_size, max_len, n_channels = X.shape
        mask = torch.ones_like(X)
        
        if pattern == "random":
            # 随机缺失
            missing_indices = torch.rand_like(X) < missing_rate
            mask[missing_indices] = 0
            
        elif pattern == "block":
            # 块状缺失
            for b in range(batch_size):
                for c in range(n_channels):
                    # 随机选择缺失块的起始位置和长度
                    block_length = int(max_len * missing_rate)
                    start_idx = torch.randint(0, max_len - block_length + 1, (1,)).item()
                    mask[b, start_idx:start_idx + block_length, c] = 0
                    
            
        return mask
    
    def apply_missing_pattern(
        self, 
        X: torch.Tensor, 
        mask: torch.Tensor, 
        fill_value: float = 0.0
    ) -> torch.Tensor:
        """
        根据掩码应用缺失模式
        
        Args:
            X: 原始时间序列
            mask: 缺失值掩码
            fill_value: 填充值
            
        Returns:
            X_missing: 带缺失值的时间序列
        """
        X_missing = X.clone()
        X_missing[mask == 0] = fill_value
        return X_missing
    
    def frequency_domain_purification(
        self, 
        X_corrupted: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        在频域中进行扩散净化
        
        Args:
            X_corrupted: 带缺失值的时间序列
            mask: 观测值掩码
            
        Returns:
            X_purified: 净化后的时间序列
        """
        self.score_model.eval()
        
        with torch.no_grad():
            # 1. 将时域缺失数据转换到频域进行处理
            # apply_missing_pattern返回的总是时域数据，所以总是需要dft
            X_freq = dft(X_corrupted)
            
            # 2. 添加噪声
            # 计算噪声时间步对应的参数
            t = torch.full(
                (X_freq.shape[0],), 
                self.noise_level, 
                device=X_freq.device
            )
            
            # 生成标准高斯噪声
            z = torch.randn_like(X_freq)
            
            # 获取边际概率参数（与训练时完全一致）
            _, std = self.noise_scheduler.marginal_prob(X_freq, t)
            
            # 按照训练时的标准方式缩放噪声
            std_matrix = torch.diag_embed(std)  # (batch_size, max_len, max_len)
            noise = torch.matmul(std_matrix, z)  # 与losses.py中完全一致的噪声缩放
            
            # 调用SDE的add_noise方法（与训练时完全一致）
            X_noisy = self.noise_scheduler.add_noise(
                original_samples=X_freq,
                noise=noise,
                timesteps=t
            )
            
            # 3. 反向扩散净化过程
            X_current = X_noisy
            
            # 选择从noise_level开始的时间步
            start_idx = int(self.noise_level * len(self.noise_scheduler.timesteps))
            timesteps_subset = self.noise_scheduler.timesteps[start_idx:]
            
            for timestep in tqdm(timesteps_subset, desc="Frequency Purification"):
                # 创建批次
                t_batch = torch.full(
                    (X_current.shape[0],),
                    timestep.item(),
                    device=X_current.device
                )
                
                batch = DiffusableBatch(X=X_current, y=None, timesteps=t_batch)
                
                # 预测分数
                score = self.score_model(batch)
                
                # 执行反向扩散步骤
                output = self.noise_scheduler.step(
                    model_output=score,
                    timestep=timestep.item(),
                    sample=X_current
                )
                
                X_current = output.prev_sample
            
            # 4. 转换回时域
            X_purified = idft(X_current.cpu()).to(X_current.device)
            
            # 5. 如果需要，保持观测值不变
            if self.preserve_observed and mask is not None:
                X_purified = X_purified * (1 - mask) + X_corrupted * mask
            
            return X_purified
    
    def iterative_imputation(
        self,
        X_missing: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        迭代补全过程
        
        Args:
            X_missing: 带缺失值的时间序列
            mask: 观测值掩码
            
        Returns:
            X_imputed: 补全后的时间序列
        """
        X_current = X_missing.clone()
        
        for iteration in range(self.max_iterations):
            print(f"Imputation iteration {iteration + 1}/{self.max_iterations}")
            
            # 频域净化
            X_purified = self.frequency_domain_purification(X_current, mask)
            
            # 更新当前估计（只更新缺失值部分）
            X_current = X_current * mask + X_purified * (1 - mask)
            
        return X_current
    
    def impute(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        missing_rate: float = 0.2,
        missing_pattern: str = "random"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        主要的补全接口
        
        Args:
            X: 完整的时间序列（用于评估）或带缺失值的时间序列
            mask: 可选的缺失值掩码，如果提供则直接使用
            missing_rate: 如果mask为None，则按此比例创建缺失值
            missing_pattern: 缺失模式
            
        Returns:
            X_imputed: 补全后的时间序列
            mask: 使用的缺失值掩码
        """
        # 移动到正确的设备
        X = X.to(self.score_model.device)
        
        if mask is None:
            # 创建缺失值掩码
            mask = self.create_missing_mask(X, missing_rate, missing_pattern)
            # 应用缺失模式
            X_missing = self.apply_missing_pattern(X, mask, fill_value=0.0)
        else:
            mask = mask.to(self.score_model.device)
            X_missing = X.clone()
            X_missing[mask == 0] = 0.0
        
        # 执行补全
        X_imputed = self.iterative_imputation(X_missing, mask)
        
        return X_imputed, mask


def evaluate_imputation_performance(
    X_true: torch.Tensor,
    X_imputed: torch.Tensor,
    mask: torch.Tensor
) -> dict:
    """
    评估补全性能
    
    Args:
        X_true: 真实值
        X_imputed: 补全值
        mask: 缺失值掩码（1表示观测值，0表示缺失值）
        
    Returns:
        metrics: 评估指标字典
    """
    # 只在缺失值位置计算误差
    missing_mask = (mask == 0)
    
    if missing_mask.sum() == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
    
    # 提取缺失位置的真实值和预测值
    true_missing = X_true[missing_mask]
    pred_missing = X_imputed[missing_mask]
    
    # 计算指标
    mse = torch.mean((true_missing - pred_missing) ** 2).item()
    mae = torch.mean(torch.abs(true_missing - pred_missing)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "missing_rate": (1 - mask.float().mean()).item()
    } 