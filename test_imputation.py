#!/usr/bin/env python3
"""
简单的频域扩散补全测试脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
import sys
sys.path.append('src')

from fdiff.utils.imputation import FrequencyDomainImputer, evaluate_imputation_performance
from fdiff.utils.fourier import dft, idft
from fdiff.schedulers.sde import VEScheduler
from fdiff.models.score_models import ScoreModule


def create_synthetic_data(batch_size=8, max_len=100, n_channels=1):
    """创建合成时间序列数据用于测试"""
    t = torch.linspace(0, 4*np.pi, max_len)
    
    # 创建包含多个频率分量的信号
    signals = []
    for i in range(batch_size):
        # 基础正弦波
        signal = torch.sin(t + i * 0.1)
        # 添加高频分量
        signal += 0.3 * torch.sin(3 * t + i * 0.2)
        # 添加低频趋势
        signal += 0.2 * torch.sin(0.5 * t + i * 0.3)
        # 添加噪声
        signal += 0.1 * torch.randn_like(t)
        
        signals.append(signal.unsqueeze(-1))  # 添加通道维度
    
    X = torch.stack(signals, dim=0)
    return X


def create_mock_score_model(max_len, n_channels):
    """创建一个模拟的分数模型用于测试"""
    
    class MockScoreModel:
        def __init__(self, max_len, n_channels):
            self.max_len = max_len
            self.n_channels = n_channels
            self.device = torch.device("cpu")
            
            # 创建噪声调度器
            self.noise_scheduler = VEScheduler(
                sigma_min=0.01,
                sigma_max=50.0,
                fourier_noise_scaling=True
            )
            self.noise_scheduler.set_noise_scaling(max_len)
            
        def eval(self):
            pass
            
        def to(self, device):
            self.device = device
            return self
            
        def __call__(self, batch):
            # 简单的模拟分数函数：返回输入的负值（简单的去噪）
            X = batch.X
            # 添加一些随机性来模拟真实的分数函数
            score = -0.1 * X + 0.05 * torch.randn_like(X)
            return score
    
    return MockScoreModel(max_len, n_channels)


def test_missing_mask_creation():
    """测试缺失值掩码创建"""
    print("=== 测试缺失值掩码创建 ===")
    
    # 创建测试数据
    X = create_synthetic_data(batch_size=2, max_len=50, n_channels=2)
    print(f"测试数据形状: {X.shape}")
    
    # 创建模拟模型
    mock_model = create_mock_score_model(X.shape[1], X.shape[2])
    imputer = FrequencyDomainImputer(mock_model)
    
    # 测试不同的缺失模式
    patterns = ["random", "block", "channel"]
    missing_rate = 0.3
    
    for pattern in patterns:
        mask = imputer.create_missing_mask(X, missing_rate, pattern)
        actual_missing_rate = 1 - mask.float().mean().item()
        
        print(f"{pattern.capitalize()} 模式:")
        print(f"  目标缺失率: {missing_rate:.1%}")
        print(f"  实际缺失率: {actual_missing_rate:.1%}")
        print(f"  掩码形状: {mask.shape}")
        print()


def test_frequency_domain_operations():
    """测试频域操作"""
    print("=== 测试频域操作 ===")
    
    # 创建测试数据
    X = create_synthetic_data(batch_size=1, max_len=64, n_channels=1)
    print(f"原始数据形状: {X.shape}")
    print(f"原始数据范围: [{X.min():.3f}, {X.max():.3f}]")
    
    # 频域变换
    X_freq = dft(X)
    print(f"频域数据形状: {X_freq.shape}")
    print(f"频域数据范围: [{X_freq.min():.3f}, {X_freq.max():.3f}]")
    
    # 逆变换
    X_reconstructed = idft(X_freq)
    print(f"重构数据形状: {X_reconstructed.shape}")
    print(f"重构数据范围: [{X_reconstructed.min():.3f}, {X_reconstructed.max():.3f}]")
    
    # 检查重构误差
    reconstruction_error = torch.mean((X - X_reconstructed) ** 2).item()
    print(f"重构误差 (MSE): {reconstruction_error:.8f}")
    
    if reconstruction_error < 1e-6:
        print("✓ 频域变换测试通过")
    else:
        print("✗ 频域变换测试失败")
    print()


def test_imputation_pipeline():
    """测试完整的补全流程"""
    print("=== 测试补全流程 ===")
    
    # 创建测试数据
    X_true = create_synthetic_data(batch_size=2, max_len=64, n_channels=1)
    print(f"真实数据形状: {X_true.shape}")
    
    # 创建模拟模型
    mock_model = create_mock_score_model(X_true.shape[1], X_true.shape[2])
    
    # 创建补全器
    imputer = FrequencyDomainImputer(
        score_model=mock_model,
        diffusion_steps=10,  # 使用较少步数进行快速测试
        noise_level=0.1,
        preserve_observed=True,
        max_iterations=1
    )
    
    # 执行补全
    print("执行补全...")
    X_imputed, mask = imputer.impute(
        X_true,
        missing_rate=0.2,
        missing_pattern="random"
    )
    
    print(f"补全数据形状: {X_imputed.shape}")
    print(f"掩码形状: {mask.shape}")
    
    # 评估性能
    metrics = evaluate_imputation_performance(X_true, X_imputed, mask)
    
    print("性能指标:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.6f}")
    
    print("✓ 补全流程测试完成")
    print()


def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化 ===")
    
    # 创建测试数据
    X_true = create_synthetic_data(batch_size=1, max_len=100, n_channels=1)
    
    # 创建模拟模型
    mock_model = create_mock_score_model(X_true.shape[1], X_true.shape[2])
    imputer = FrequencyDomainImputer(mock_model, diffusion_steps=5)
    
    # 执行补全
    X_imputed, mask = imputer.impute(
        X_true,
        missing_rate=0.3,
        missing_pattern="random"
    )
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 提取数据用于绘图
    x_true = X_true[0, :, 0].numpy()
    x_imputed = X_imputed[0, :, 0].numpy()
    mask_1d = mask[0, :, 0].numpy()
    
    # 创建带缺失值的信号
    x_missing = x_true.copy()
    x_missing[mask_1d == 0] = 0
    
    # 子图1: 时域比较
    time_steps = np.arange(len(x_true))
    axes[0, 0].plot(time_steps, x_true, 'b-', label='真实值', alpha=0.8)
    axes[0, 0].plot(time_steps, x_missing, 'gray', label='带缺失值', alpha=0.6)
    axes[0, 0].plot(time_steps, x_imputed, 'r--', label='补全值', alpha=0.8)
    
    missing_indices = np.where(mask_1d == 0)[0]
    if len(missing_indices) > 0:
        axes[0, 0].scatter(missing_indices, x_imputed[missing_indices], 
                          c='red', s=20, label='补全点', zorder=5)
    
    axes[0, 0].set_title('时域补全结果')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 误差分析
    error = np.abs(x_true - x_imputed)
    axes[0, 1].plot(time_steps, error, 'g-', alpha=0.8)
    if len(missing_indices) > 0:
        axes[0, 1].scatter(missing_indices, error[missing_indices], 
                          c='red', s=20, zorder=5)
    axes[0, 1].set_title('绝对误差')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('|真实值 - 补全值|')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 频域比较
    X_true_freq = dft(X_true).squeeze().numpy()
    X_imputed_freq = dft(X_imputed).squeeze().numpy()
    
    freq_bins = np.arange(len(X_true_freq))
    axes[1, 0].plot(freq_bins, X_true_freq[:, 0], 'b-', label='真实值(频域)', alpha=0.8)
    axes[1, 0].plot(freq_bins, X_imputed_freq[:, 0], 'r--', label='补全值(频域)', alpha=0.8)
    axes[1, 0].set_title('频域比较')
    axes[1, 0].set_xlabel('频率分量')
    axes[1, 0].set_ylabel('幅度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 缺失模式
    axes[1, 1].plot(time_steps, mask_1d, 'k-', linewidth=2)
    axes[1, 1].fill_between(time_steps, 0, mask_1d, alpha=0.3, color='blue', label='观测值')
    axes[1, 1].fill_between(time_steps, 0, 1-mask_1d, alpha=0.3, color='red', label='缺失值')
    axes[1, 1].set_title(f'缺失模式 (缺失率: {(1-mask_1d.mean()):.1%})')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('掩码值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "imputation_test.png", dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_dir / 'imputation_test.png'}")
    
    # 显示图像（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    print("✓ 可视化测试完成")
    print()


def main():
    """主测试函数"""
    print("开始频域扩散补全测试...\n")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 运行各项测试
        test_missing_mask_creation()
        test_frequency_domain_operations()
        test_imputation_pipeline()
        test_visualization()
        
        print("🎉 所有测试完成！")
        print("\n下一步:")
        print("1. 训练真实的频域扩散模型:")
        print("   python cmd/train.py fourier_transform=true datamodule=ecg")
        print("2. 使用真实模型进行补全:")
        print("   python cmd/impute.py model_id=YOUR_MODEL_ID missing_rate=0.2")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 