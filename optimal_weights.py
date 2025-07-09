import numpy as np
import matplotlib.pyplot as plt

def optimal_weights(r1, r2, r12):
    """
    计算最优权重，使得加权信号与目标的相关系数最大
    
    参数:
    r1: 信号x1与目标y的相关系数
    r2: 信号x2与目标y的相关系数  
    r12: 信号x1与x2的相关系数
    
    返回:
    w1_opt, w2_opt: 最优权重
    max_corr: 最大可能的相关系数
    """
    
    # 检查输入有效性
    if abs(r12) >= 1:
        raise ValueError("r12必须在(-1, 1)范围内")
    
    # 计算最优权重（未归一化）
    denominator = 1 - r12**2
    w1_unnorm = (r1 - r12 * r2) / denominator
    w2_unnorm = (r2 - r12 * r1) / denominator
    
    # 计算最大相关系数
    numerator = r1**2 + r2**2 - 2*r1*r2*r12
    max_corr = np.sqrt(numerator / denominator)
    
    # 归一化权重（可选，使权重平方和为1）
    norm = np.sqrt(w1_unnorm**2 + w2_unnorm**2)
    w1_norm = w1_unnorm / norm
    w2_norm = w2_unnorm / norm
    
    return {
        'w1_optimal': w1_unnorm,
        'w2_optimal': w2_unnorm,
        'w1_normalized': w1_norm,
        'w2_normalized': w2_norm,
        'max_correlation': max_corr,
        'weight_ratio': w1_unnorm / w2_unnorm if w2_unnorm != 0 else float('inf')
    }

def analyze_weight_sensitivity(r1, r2, r12_range=None):
    """分析权重对r12的敏感性"""
    if r12_range is None:
        r12_range = np.linspace(-0.9, 0.9, 100)
    
    results = []
    for r12 in r12_range:
        try:
            result = optimal_weights(r1, r2, r12)
            results.append({
                'r12': r12,
                'w1': result['w1_optimal'],
                'w2': result['w2_optimal'],
                'max_corr': result['max_correlation']
            })
        except:
            continue
    
    return results

# 示例使用
if __name__ == "__main__":
    # 示例1：两个信号都与目标正相关
    print("=== 示例1：r1=0.12, r2=0.13, r12=0.94 ===")
    result1 = optimal_weights(r1=0.12, r2=0.13, r12=0.94)
    print(f"最优权重: w1={result1['w1_optimal']:.4f}, w2={result1['w2_optimal']:.4f}")
    print(f"权重比例: w1:w2 = {result1['weight_ratio']:.2f}:1")
    print(f"最大相关系数: {result1['max_correlation']:.4f}")
 