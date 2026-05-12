# 训练结果可视化指南

## 概述
已为您添加了完整的训练结果可视化功能。修改后的训练脚本会自动保存所有训练指标（损失、准确率、F-Score），两个可视化脚本可用于展示这些指标。

## 修改内容

### 1. `train_fourier.py` 修改
- 添加了 `history` 字典来记录每个epoch的所有指标
- 收集以下数据：
  - 训练/验证/测试的损失 (Loss)
  - 训练/验证/测试的准确率 (Accuracy)
  - 训练/验证/测试的F-Score
- 在训练完成后将历史数据保存为JSON文件：`{model_name}_history.json`

### 2. 可视化脚本

#### 方案1: `visualize_training.py` (交互式)
- 需要图形界面支持
- 可实时显示和交互式查看
- 支持缩放、平移等功能

**使用方法：**
```bash
conda activate gs-mcc
python visualize_training.py --model-name concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal
```

**选项说明：**
```bash
--history FILE_PATH          # 直接指定历史JSON文件路径
--model-name NAME            # 模型名称（默认：concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal）
--save-dir DIR              # 历史文件所在目录（默认：./saved/IEMOCAP/）
--output-dir DIR            # 保存输出图像的目录（默认：./visualization/）
--detail                    # 生成单个指标的详细分析图表
```

#### 方案2: `visualize_training_static.py` (推荐用于服务器/自动化)
- 不需要图形界面
- 生成高质量PNG图像
- 适合无头服务器环境
- **推荐使用此方案**

**使用方法：**
```bash
conda activate gs-mcc
python visualize_training_static.py --model-name concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal --detail
```

**输出示例：**
```
📂 加载历史文件: ./saved/IEMOCAP/concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal_history.json
📊 生成训练过程可视化...
✓ 综合图表已保存: ./visualization/training_overview.png
📈 生成详细指标分析...
✓ Loss详细分析已保存: ./visualization/loss_analysis.png
✓ Accuracy详细分析已保存: ./visualization/accuracy_analysis.png
✓ F-Score详细分析已保存: ./visualization/fscore_analysis.png
✅ 完成！所有图表已保存到: ./visualization/
```

## 工作流程

### 训练阶段
1. 运行修改后的训练脚本
   ```bash
   python train_fourier.py --Dataset IEMOCAP --epochs 250 --batch-size 32
   ```

2. 训练完成后，历史数据自动保存为：
   ```
   saved/IEMOCAP/{model_name}_history.json
   ```

### 可视化阶段
3. 运行可视化脚本生成图表
   ```bash
   python visualize_training_static.py --model-name {model_name} --detail
   ```

4. 查看生成的图表
   ```
   visualization/training_overview.png         # 综合4图表
   visualization/loss_analysis.png             # 损失详细分析
   visualization/accuracy_analysis.png         # 准确率详细分析
   visualization/fscore_analysis.png           # F-Score详细分析
   ```

## 输出图表说明

### 综合图表 (training_overview.png)
包含4个子图：
1. **损失曲线** - 显示训练/验证/测试的损失趋势
2. **准确率曲线** - 显示三个阶段的准确率变化
3. **F-Score曲线** - 显示加权F-Score
4. **统计摘要** - 显示最佳性能、最终性能和训练轮次

### 详细分析图表
- `loss_analysis.png` - 大尺寸损失分析
- `accuracy_analysis.png` - 大尺寸准确率分析
- `fscore_analysis.png` - 大尺寸F-Score分析

## 性能指标说明

- **Loss（损失）** - 越低越好，用于判断模型是否过拟合
- **Accuracy（准确率）** - 百分比，越高越好
- **F-Score** - 加权平均F-Score，综合考虑precision和recall

## 示例命令

### 基础用法 (生成综合图表)
```bash
python visualize_training_static.py
```

### 详细分析 (生成所有图表)
```bash
python visualize_training_static.py --detail
```

### 自定义输出目录
```bash
python visualize_training_static.py --output-dir ./my_results/ --detail
```

### 指定特定的历史文件
```bash
python visualize_training_static.py --history "./saved/IEMOCAP/my_model_history.json"
```

## 技术细节

### JSON历史文件格式
```json
{
    "epochs": [1, 2, 3, ..., 250],
    "train_loss": [3.45, 3.23, ..., 1.02],
    "train_acc": [25.5, 28.3, ..., 85.2],
    "train_fscore": [24.1, 27.8, ..., 84.5],
    "valid_loss": [3.52, 3.35, ..., 1.15],
    "valid_acc": [24.8, 27.1, ..., 83.1],
    "valid_fscore": [23.5, 26.9, ..., 82.3],
    "test_loss": [3.58, 3.42, ..., 1.20],
    "test_acc": [24.2, 26.5, ..., 82.5],
    "test_fscore": [23.0, 26.1, ..., 81.8]
}
```

### 使用的库
- `matplotlib` - 数据可视化
- `json` - 历史数据存储
- `numpy` - 数值计算

## 故障排除

### 找不到历史文件
确保：
1. 训练脚本已完成运行
2. 使用了正确的 `--model-name`
3. 或通过 `--history` 参数指定完整路径

### 图表显示乱码
如遇到中文乱码，可以：
1. 在脚本中修改字体设置
2. 或运行时指定字体环境变量

### 内存不足
减少DPI或删除 `--detail` 参数只生成综合图表

## 后续改进建议
- 支持对比多个模型的训练历史
- 生成HTML交互式报告
- 添加混淆矩阵可视化
- 集成TensorBoard支持
