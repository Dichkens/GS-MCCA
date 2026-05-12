"""
简化版可视化脚本 - 生成静态PNG图像，无需图形界面
使用方法：
  python visualize_training_static.py --model-name your_model_name
"""

import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # 使用非图形界面后端
import matplotlib.pyplot as plt
import numpy as np

def load_history(history_file):
    """加载训练历史JSON文件"""
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def plot_metrics(history, save_path):
    """绘制训练、验证和测试指标"""
    epochs = history['epochs']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('情绪识别模型训练过程可视化', fontsize=16, fontweight='bold')
    
    # 损失曲线
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_loss'], 's-', label='Valid Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_loss'], '^-', label='Test Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('损失函数 (Loss)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_acc'], 's-', label='Valid Acc', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('准确率 (Accuracy)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # F-Score曲线
    ax = axes[1, 0]
    ax.plot(epochs, history['train_fscore'], 'o-', label='Train F-Score', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_fscore'], 's-', label='Valid F-Score', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_fscore'], '^-', label='Test F-Score', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F-Score (%)', fontsize=11)
    ax.set_title('F-Score (加权平均)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    
    best_test_fscore_idx = np.argmax(history['test_fscore'])
    best_test_epoch = history['epochs'][best_test_fscore_idx]
    
    stats_text = f"""训练统计摘要

最佳测试 F-Score:  {max(history['test_fscore']):.2f}% (Epoch {best_test_epoch})
最佳测试准确率:    {max(history['test_acc']):.2f}%
最小测试损失:      {min(history['test_loss']):.4f}

最终测试 F-Score:  {history['test_fscore'][-1]:.2f}%
最终测试准确率:    {history['test_acc'][-1]:.2f}%
最终测试损失:      {history['test_loss'][-1]:.4f}

总训练轮次:        {len(epochs)}
"""
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 综合图表已保存: {save_path}")
    plt.close()

def plot_individual_metric(history, metric_type, save_path):
    """绘制单个指标的详细分析"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if metric_type == 'fscore':
        train_data = history['train_fscore']
        valid_data = history['valid_fscore']
        test_data = history['test_fscore']
        ylabel = 'F-Score (%)'
        title = 'F-Score 详细分析'
        color_name = '#FF6B6B'
    elif metric_type == 'accuracy':
        train_data = history['train_acc']
        valid_data = history['valid_acc']
        test_data = history['test_acc']
        ylabel = 'Accuracy (%)'
        title = '准确率详细分析'
        color_name = '#4ECDC4'
    elif metric_type == 'loss':
        train_data = history['train_loss']
        valid_data = history['valid_loss']
        test_data = history['test_loss']
        ylabel = 'Loss'
        title = '损失函数详细分析'
        color_name = '#45B7D1'
    
    epochs = history['epochs']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_data, 'o-', label='Train', linewidth=2.5, markersize=6, color='#1f77b4')
    ax.plot(epochs, valid_data, 's-', label='Validation', linewidth=2.5, markersize=6, color='#ff7f0e')
    ax.plot(epochs, test_data, '^-', label='Test', linewidth=2.5, markersize=6, color='#2ca02c')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {title}已保存: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='训练历史可视化 - 生成静态PNG图像')
    parser.add_argument('--history', type=str, help='历史JSON文件的完整路径')
    parser.add_argument('--model-name', type=str, 
                        default='concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal',
                        help='模型名称（用于自动查找历史文件）')
    parser.add_argument('--save-dir', type=str, default='./saved/IEMOCAP/',
                        help='历史文件所在目录')
    parser.add_argument('--output-dir', type=str, default='./visualization/',
                        help='保存可视化图像的目录')
    parser.add_argument('--detail', action='store_true', 
                        help='生成单个指标的详细分析图表')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定历史文件路径
    if args.history:
        history_file = args.history
    else:
        history_file = os.path.join(args.save_dir, f'{args.model_name}_history.json')
    
    print(f"📂 加载历史文件: {history_file}")
    history = load_history(history_file)
    
    # 生成综合可视化
    print("\n📊 生成训练过程可视化...")
    main_plot_path = os.path.join(args.output_dir, 'training_overview.png')
    plot_metrics(history, main_plot_path)
    
    # 生成详细分析图表
    if args.detail:
        print("\n📈 生成详细指标分析...")
        for metric, metric_name in [('loss', 'Loss'), ('accuracy', 'Accuracy'), ('fscore', 'F-Score')]:
            detail_path = os.path.join(args.output_dir, f'{metric}_analysis.png')
            plot_individual_metric(history, metric, detail_path)
    
    print(f"\n✅ 完成！所有图表已保存到: {args.output_dir}")

if __name__ == '__main__':
    main()
