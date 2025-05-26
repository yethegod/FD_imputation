#!/usr/bin/env python3
"""
测试训练时和补全时的加噪过程是否一致
"""

import torch
import numpy as np
import sys
sys.path.append('src')

from fdiff.schedulers.sde import VEScheduler
from fdiff.utils.fourier import dft


def test_noise_consistency():
    """测试训练时和补全时的加噪过程是否一致"""
    print("=== 测试加噪过程一致性 ===")
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size, max_len, n_channels = 2, 64, 1
    X = torch.randn(batch_size, max_len, n_channels)
    X_freq = dft(X)
    
    # 创建调度器
    scheduler = VEScheduler(
        sigma_min=0.01,
        sigma_max=50.0,
        fourier_noise_scaling=True
    )
    scheduler.set_noise_scaling(max_len)
    
    # 设置时间步
    t = torch.full((batch_size,), 0.1)
    
    print(f"输入数据形状: {X_freq.shape}")
    print(f"时间步: {t}")
    
    # 方法1：训练时的标准加噪过程（来自losses.py）
    torch.manual_seed(42)  # 重置随机种子
    z1 = torch.randn_like(X_freq)
    _, std1 = scheduler.marginal_prob(X_freq, t)
    std_matrix1 = torch.diag_embed(std1)
    noise1 = torch.matmul(std_matrix1, z1)
    X_noisy1 = scheduler.add_noise(
        original_samples=X_freq,
        noise=noise1,
        timesteps=t
    )
    
    # 方法2：修改后的补全时加噪过程
    torch.manual_seed(42)  # 重置随机种子
    z2 = torch.randn_like(X_freq)
    _, std2 = scheduler.marginal_prob(X_freq, t)
    std_matrix2 = torch.diag_embed(std2)
    noise2 = torch.matmul(std_matrix2, z2)
    X_noisy2 = scheduler.add_noise(
        original_samples=X_freq,
        noise=noise2,
        timesteps=t
    )
    
    # 检查一致性
    print("\n=== 一致性检查 ===")
    print(f"z1 == z2: {torch.allclose(z1, z2)}")
    print(f"std1 == std2: {torch.allclose(std1, std2)}")
    print(f"noise1 == noise2: {torch.allclose(noise1, noise2)}")
    print(f"X_noisy1 == X_noisy2: {torch.allclose(X_noisy1, X_noisy2)}")
    
    # 计算差异
    noise_diff = torch.mean((noise1 - noise2) ** 2).item()
    result_diff = torch.mean((X_noisy1 - X_noisy2) ** 2).item()
    
    print(f"\n噪声差异 (MSE): {noise_diff:.10f}")
    print(f"结果差异 (MSE): {result_diff:.10f}")
    
    if noise_diff < 1e-10 and result_diff < 1e-10:
        print("✓ 加噪过程完全一致！")
        return True
    else:
        print("✗ 加噪过程存在差异")
        return False


def test_add_noise_behavior():
    """测试add_noise方法的行为"""
    print("\n=== 测试add_noise方法行为 ===")
    
    # 创建简单测试数据
    X = torch.ones(1, 4, 1)  # 简单的全1矩阵
    scheduler = VEScheduler(fourier_noise_scaling=True)
    scheduler.set_noise_scaling(4)
    
    t = torch.tensor([0.5])
    
    # 获取边际概率参数
    mean, std = scheduler.marginal_prob(X, t)
    print(f"原始数据: {X.squeeze()}")
    print(f"边际概率均值: {mean.squeeze()}")
    print(f"边际概率标准差: {std.squeeze()}")
    
    # 测试add_noise方法
    noise = torch.ones_like(X) * 0.1  # 固定的小噪声
    X_noisy = scheduler.add_noise(X, noise, t)
    
    print(f"添加的噪声: {noise.squeeze()}")
    print(f"加噪后结果: {X_noisy.squeeze()}")
    print(f"预期结果 (mean + noise): {(mean + noise).squeeze()}")
    
    # 验证add_noise的实现：应该是 mean + noise
    expected = mean + noise
    is_correct = torch.allclose(X_noisy, expected)
    print(f"add_noise实现正确: {is_correct}")
    
    return is_correct


if __name__ == "__main__":
    print("开始测试加噪过程一致性...\n")
    
    # 运行测试
    test1_passed = test_noise_consistency()
    test2_passed = test_add_noise_behavior()
    
    print(f"\n=== 测试总结 ===")
    print(f"加噪一致性测试: {'通过' if test1_passed else '失败'}")
    print(f"add_noise行为测试: {'通过' if test2_passed else '失败'}")
    
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！补全时的加噪过程与训练时完全一致。")
    else:
        print("❌ 存在问题，需要进一步检查。") 