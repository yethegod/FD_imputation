#!/usr/bin/env python3
"""
基于频域扩散的时间序列缺失值补全脚本

使用方法:
python cmd/impute.py model_id=YOUR_MODEL_ID missing_rate=0.2 missing_pattern=random
"""

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from omegaconf import DictConfig
import logging

from fdiff.utils.imputation import FrequencyDomainImputer, evaluate_imputation_performance
from fdiff.models.score_models import ScoreModule
from fdiff.dataloaders.datamodules import *
from fdiff.utils.fourier import dft, idft


@hydra.main(version_base=None, config_path="conf", config_name="impute")
def main(cfg: DictConfig) -> None:
    """主函数"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 检查模型ID
    if not cfg.model_id:
        raise ValueError("Please provide model_id parameter")
    
    # 加载预训练模型
    model_path = Path("lightning_logs") / cfg.model_id / "checkpoints"
    checkpoint_files = list(model_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    
    checkpoint_path = checkpoint_files[0]  # 使用第一个检查点
    logger.info(f"Loading model: {checkpoint_path}")
    
    # 加载模型
    score_model = ScoreModule.load_from_checkpoint(checkpoint_path)
    score_model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_model = score_model.to(device)
    
    # 加载数据模块
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    
    # 获取测试数据
    test_loader = datamodule.test_dataloader()
    
    # 创建补全器
    imputer = FrequencyDomainImputer(
        score_model=score_model,
        diffusion_steps=cfg.diffusion_steps,
        noise_level=cfg.noise_level,
        preserve_observed=cfg.preserve_observed,
        max_iterations=cfg.max_iterations
    )
    
    logger.info(f"Starting imputation experiment:")
    logger.info(f"  - Missing rate: {cfg.missing_rate}")
    logger.info(f"  - Missing pattern: {cfg.missing_pattern}")
    logger.info(f"  - Noise level: {cfg.noise_level}")
    logger.info(f"  - Diffusion steps: {cfg.diffusion_steps}")
    
    # 收集结果
    all_metrics = []
    all_samples = []
    
    # 处理测试批次
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= cfg.max_batches:
            break
            
        X_true = batch.X.to(device)
        
        logger.info(f"Processing batch {batch_idx + 1}/{min(len(test_loader), cfg.max_batches)}")
        
        # 执行补全
        X_imputed, mask = imputer.impute(
            X_true,
            missing_rate=cfg.missing_rate,
            missing_pattern=cfg.missing_pattern,
            input_is_frequency_domain=cfg.fourier_transform  # 根据配置确定输入域
        )
        
        # 评估性能
        metrics = evaluate_imputation_performance(X_true, X_imputed, mask)
        all_metrics.append(metrics)
        
        logger.info(f"  MSE: {metrics['mse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        
        # 保存样本用于可视化
        if batch_idx < cfg.num_visualization_samples:
            all_samples.append({
                'X_true': X_true.cpu(),
                'X_imputed': X_imputed.cpu(),
                'mask': mask.cpu()
            })
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))  # 确保转换为 Python float
    
    logger.info("\n=== Average Performance Metrics ===")
    for key, value in avg_metrics.items():
        logger.info(f"{key.upper()}: {value:.6f}")
    
    # 保存结果
    results_dir = Path("imputation_results") / cfg.model_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标为YAML格式
    with open(results_dir / "impute_result.yaml", "w", encoding="utf-8") as f:
        yaml.dump(avg_metrics, f, default_flow_style=False, allow_unicode=False)
    
    # 保存样本数据（保持原来的pt格式）
    torch.save(all_samples, results_dir / "samples.pt")
    
    # 创建可视化
    if cfg.create_visualizations and all_samples:
        create_imputation_visualizations(all_samples, results_dir, cfg)
    
    logger.info(f"Results saved to: {results_dir}")


def create_imputation_visualizations(samples, results_dir, cfg):
    """创建补全结果的可视化"""
    
    for sample_idx, sample in enumerate(samples[:3]):  # 只可视化前3个样本
        X_true = sample['X_true']
        X_imputed = sample['X_imputed']
        mask = sample['mask']
        
        # 选择第一个样本的第一个通道进行可视化
        x_true = X_true[0, :, 0].numpy()
        x_imputed = X_imputed[0, :, 0].numpy()
        mask_1d = mask[0, :, 0].numpy()
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 子图1: 原始vs补全
        plt.subplot(2, 2, 1)
        time_steps = np.arange(len(x_true))
        plt.plot(time_steps, x_true, 'b-', label='Ground Truth', alpha=0.7)
        plt.plot(time_steps, x_imputed, 'r--', label='Imputed', alpha=0.7)
        
        # 标记缺失值位置
        missing_indices = np.where(mask_1d == 0)[0]
        if len(missing_indices) > 0:
            plt.scatter(missing_indices, x_imputed[missing_indices], 
                       c='red', s=20, label='Imputed Points', zorder=5)
        
        plt.title('Time Domain Imputation Results')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 误差分析
        plt.subplot(2, 2, 2)
        error = np.abs(x_true - x_imputed)
        plt.plot(time_steps, error, 'g-', alpha=0.7)
        if len(missing_indices) > 0:
            plt.scatter(missing_indices, error[missing_indices], 
                       c='red', s=20, zorder=5)
        plt.title('Absolute Error')
        plt.xlabel('Time Step')
        plt.ylabel('|Ground Truth - Imputed|')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 频域比较
        plt.subplot(2, 2, 3)
        X_true_freq = dft(X_true[0:1]).squeeze(0).numpy()  # 只移除批次维度
        X_imputed_freq = dft(X_imputed[0:1]).squeeze(0).numpy()  # 只移除批次维度
        
        # 如果只有一个通道，确保是2维数组
        if X_true_freq.ndim == 1:
            X_true_freq = X_true_freq[:, None]
            X_imputed_freq = X_imputed_freq[:, None]
        
        freq_bins = np.arange(X_true_freq.shape[0])
        plt.plot(freq_bins, X_true_freq[:, 0], 'b-', label='Ground Truth (Freq)', alpha=0.7)
        plt.plot(freq_bins, X_imputed_freq[:, 0], 'r--', label='Imputed (Freq)', alpha=0.7)
        plt.title('Frequency Domain Comparison')
        plt.xlabel('Frequency Component')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4: 缺失模式
        plt.subplot(2, 2, 4)
        plt.plot(time_steps, mask_1d, 'k-', linewidth=2)
        plt.fill_between(time_steps, 0, mask_1d, alpha=0.3, color='blue', label='Observed')
        plt.fill_between(time_steps, 0, 1-mask_1d, alpha=0.3, color='red', label='Missing')
        plt.title(f'Missing Pattern (Missing Rate: {(1-mask_1d.mean()):.1%})')
        plt.xlabel('Time Step')
        plt.ylabel('Mask Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f"imputation_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main() 