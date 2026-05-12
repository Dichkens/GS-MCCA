import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better display
import numpy as np

def load_history(history_file):
    """Load training history from JSON file"""
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"History file not found: {history_file}")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return history

def plot_metrics(history, save_path=None):
    """Plot training, validation, and test metrics"""
    epochs = history['epochs']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Process Visualization', fontsize=16, fontweight='bold')
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_loss'], 's-', label='Valid Loss', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_loss'], '^-', label='Test Loss', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_acc'], 's-', label='Valid Acc', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # F-Score curves
    ax = axes[1, 0]
    ax.plot(epochs, history['train_fscore'], 'o-', label='Train F-Score', linewidth=2, markersize=4)
    ax.plot(epochs, history['valid_fscore'], 's-', label='Valid F-Score', linewidth=2, markersize=4)
    ax.plot(epochs, history['test_fscore'], '^-', label='Test F-Score', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F-Score (%)', fontsize=11)
    ax.set_title('F-Score weighted', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""abstract:
    
best F-Score: {max(history['test_fscore']):.2f}%
best Test Accuracy: {max(history['test_acc']):.2f}%
minimum Test Loss: {min(history['test_loss']):.4f}

final Test F-Score: {history['test_fscore'][-1]:.2f}%
final Test Accuracy: {history['test_acc'][-1]:.2f}%
final Test Loss: {history['test_loss'][-1]:.4f}

Total Training Epochs: {len(epochs)}
"""
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_individual_metrics(history, metric_type='fscore', save_dir=None):
    """Plot individual metrics with larger detail"""
    if metric_type == 'fscore':
        train_data = history['train_fscore']
        valid_data = history['valid_fscore']
        test_data = history['test_fscore']
        ylabel = 'F-Score (%)'
        title = 'F-Score Detailed Analysis'
    elif metric_type == 'accuracy':
        train_data = history['train_acc']
        valid_data = history['valid_acc']
        test_data = history['test_acc']
        ylabel = 'Accuracy (%)'
        title = 'Accuracy Detailed Analysis'
    elif metric_type == 'loss':
        train_data = history['train_loss']
        valid_data = history['valid_loss']
        test_data = history['test_loss']
        ylabel = 'Loss'
        title = 'Loss Detailed Analysis'
    
    epochs = history['epochs']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_data, 'o-', label='Train', linewidth=2.5, markersize=6, color='#1f77b4')
    ax.plot(epochs, valid_data, 's-', label='Valid', linewidth=2.5, markersize=6, color='#ff7f0e')
    ax.plot(epochs, test_data, '^-', label='Test', linewidth=2.5, markersize=6, color='#2ca02c')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'{metric_type}_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def generate_analysis_md(history, model_name, output_path):
    """Generate a Markdown file with model analysis"""
    epochs = history['epochs']
    
    # Calculate statistics
    best_test_fscore = max(history['test_fscore'])
    best_test_acc = max(history['test_acc'])
    min_test_loss = min(history['test_loss'])
    
    final_test_fscore = history['test_fscore'][-1]
    final_test_acc = history['test_acc'][-1]
    final_test_loss = history['test_loss'][-1]
    
    total_epochs = len(epochs)
    
    # Generate Markdown content
    md_content = f"""# 模型训练分析报告

## 模型信息
- **模型名称**: {model_name}
- **训练日期**: {np.datetime64('today').astype(str)}
- **总训练轮数**: {total_epochs}

## 性能统计

### 最佳性能指标
- **最佳测试F-Score**: {best_test_fscore:.2f}%
- **最佳测试准确率**: {best_test_acc:.2f}%
- **最低测试损失**: {min_test_loss:.4f}

### 最终性能指标
- **最终测试F-Score**: {final_test_fscore:.2f}%
- **最终测试准确率**: {final_test_acc:.2f}%
- **最终测试损失**: {final_test_loss:.4f}

## 训练过程分析

### 损失曲线趋势
- 训练损失: 从 {history['train_loss'][0]:.4f} 到 {history['train_loss'][-1]:.4f}
- 验证损失: 从 {history['valid_loss'][0]:.4f} 到 {history['valid_loss'][-1]:.4f}
- 测试损失: 从 {history['test_loss'][0]:.4f} 到 {history['test_loss'][-1]:.4f}

### 准确率曲线趋势
- 训练准确率: 从 {history['train_acc'][0]:.2f}% 到 {history['train_acc'][-1]:.2f}%
- 验证准确率: 从 {history['valid_acc'][0]:.2f}% 到 {history['valid_acc'][-1]:.2f}%
- 测试准确率: 从 {history['test_acc'][0]:.2f}% 到 {history['test_acc'][-1]:.2f}%

### F-Score曲线趋势
- 训练F-Score: 从 {history['train_fscore'][0]:.2f}% 到 {history['train_fscore'][-1]:.2f}%
- 验证F-Score: 从 {history['valid_fscore'][0]:.2f}% 到 {history['valid_fscore'][-1]:.2f}%
- 测试F-Score: 从 {history['test_fscore'][0]:.2f}% 到 {history['test_fscore'][-1]:.2f}%

## 可视化图表
- ![训练概览](training_overview.png)
- ![损失分析](loss_analysis.png)
- ![准确率分析](accuracy_analysis.png)
- ![F-Score分析](fscore_analysis.png)

## 结论
该模型在训练过程中显示出[在此处添加具体分析]的性能表现。建议进一步的超参数调优或数据增强以提升性能。

---
*此报告由自动分析脚本生成*
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Analysis Markdown file generated: {output_path}")

def main():

    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--history', type=str, help='Path to history JSON file')
    parser.add_argument('--model-name', type=str, default='concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal',
                        help='Model name for automatic history file search')
    parser.add_argument('--save-dir', type=str, help='Directory where history file is saved (auto-detected from model name if not provided)')
    parser.add_argument('--output-dir', type=str, default='./visualization/',
                        help='Directory to save visualization images')
    parser.add_argument('--detail', action='store_true', help='Show detailed individual metric plots')
    
    args = parser.parse_args()
    
    # Extract dataset name from model name
    if 'IEMOCAP' in args.model_name.upper():
        dataset_name = 'IEMOCAP'
    elif 'MELD' in args.model_name.upper():
        dataset_name = 'MELD'
    else:
        dataset_name = 'Unknown'
    
    # Set save_dir if not provided
    if args.save_dir is None:
        args.save_dir = f'./saved/{dataset_name}/'
    
    # Set output directory to dataset-specific subdirectory
    args.output_dir = os.path.join(args.output_dir, dataset_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine history file path
    if args.history:
        history_file = args.history
    else:
        history_file = os.path.join(args.save_dir, f'{args.model_name}_history.json')
    
    print(f"Loading history from: {history_file}")
    history = load_history(history_file)
    
    # Main visualization
    main_plot_path = os.path.join(args.output_dir, 'training_overview.png')
    plot_metrics(history, main_plot_path)
    
    # Detailed plots if requested
    if args.detail:
        print("\nGenerating detailed metric plots...")
        for metric in ['loss', 'accuracy', 'fscore']:
            plot_individual_metrics(history, metric, args.output_dir)
    
    # Generate analysis Markdown file
    md_path = os.path.join(args.output_dir, 'model_analysis.md')
    generate_analysis_md(history, args.model_name, md_path)
    
    print(f"\nAll visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
