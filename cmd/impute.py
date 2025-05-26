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
        raise ValueError("请提供model_id参数")
    
    # 加载预训练模型
    model_path = Path("lightning_logs") / cfg.model_id / "checkpoints"
    checkpoint_files = list(model_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"在 {model_path} 中找不到检查点文件")
    
    checkpoint_path = checkpoint_files[0]  # 使用第一个检查点
    logger.info(f"加载模型: {checkpoint_path}")
    
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
    
    logger.info(f"开始补全实验:")
    logger.info(f"  - 缺失率: {cfg.missing_rate}")
    logger.info(f"  - 缺失模式: {cfg.missing_pattern}")
    logger.info(f"  - 噪声水平: {cfg.noise_level}")
    logger.info(f"  - 扩散步数: {cfg.diffusion_steps}")
    
    # 收集结果
    all_metrics = []
    all_samples = []
    
    # 处理测试批次
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= cfg.max_batches:
            break
            
        X_true = batch["X"].to(device)
        
        logger.info(f"处理批次 {batch_idx + 1}/{min(len(test_loader), cfg.max_batches)}")
        
        # 执行补全
        X_imputed, mask = imputer.impute(
            X_true,
            missing_rate=cfg.missing_rate,
            missing_pattern=cfg.missing_pattern
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
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    logger.info("\n=== 平均性能指标 ===")
    for key, value in avg_metrics.items():
        logger.info(f"{key.upper()}: {value:.6f}")
    
    # 保存结果
    results_dir = Path("imputation_results") / cfg.model_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标
    torch.save(avg_metrics, results_dir / "metrics.pt")
    torch.save(all_samples, results_dir / "samples.pt")
    
    # 创建可视化
    if cfg.create_visualizations and all_samples:
        create_imputation_visualizations(all_samples, results_dir, cfg)
    
    logger.info(f"结果已保存到: {results_dir}")


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
        plt.plot(time_steps, x_true, 'b-', label='真实值', alpha=0.7)
        plt.plot(time_steps, x_imputed, 'r--', label='补全值', alpha=0.7)
        
        # 标记缺失值位置
        missing_indices = np.where(mask_1d == 0)[0]
        if len(missing_indices) > 0:
            plt.scatter(missing_indices, x_imputed[missing_indices], 
                       c='red', s=20, label='补全点', zorder=5)
        
        plt.title('时域补全结果')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 误差分析
        plt.subplot(2, 2, 2)
        error = np.abs(x_true - x_imputed)
        plt.plot(time_steps, error, 'g-', alpha=0.7)
        if len(missing_indices) > 0:
            plt.scatter(missing_indices, error[missing_indices], 
                       c='red', s=20, zorder=5)
        plt.title('绝对误差')
        plt.xlabel('时间步')
        plt.ylabel('|真实值 - 补全值|')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 频域比较
        plt.subplot(2, 2, 3)
        X_true_freq = dft(X_true[0:1]).squeeze().numpy()
        X_imputed_freq = dft(X_imputed[0:1]).squeeze().numpy()
        
        freq_bins = np.arange(len(X_true_freq))
        plt.plot(freq_bins, X_true_freq[:, 0], 'b-', label='真实值(频域)', alpha=0.7)
        plt.plot(freq_bins, X_imputed_freq[:, 0], 'r--', label='补全值(频域)', alpha=0.7)
        plt.title('频域比较')
        plt.xlabel('频率分量')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图4: 缺失模式
        plt.subplot(2, 2, 4)
        plt.plot(time_steps, mask_1d, 'k-', linewidth=2)
        plt.fill_between(time_steps, 0, mask_1d, alpha=0.3, color='blue', label='观测值')
        plt.fill_between(time_steps, 0, 1-mask_1d, alpha=0.3, color='red', label='缺失值')
        plt.title(f'缺失模式 (缺失率: {(1-mask_1d.mean()):.1%})')
        plt.xlabel('时间步')
        plt.ylabel('掩码值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f"imputation_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main() 