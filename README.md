# GS-MCC: Multimodal Emotion Recognition with Graph Spectrum and Fourier GNN

> 这是一个基于跨模态注意力的多模态情感识别模型，结合图谱谱特征、傅里叶图神经网络和多模态融合优化。支持 IEMOCAP 与 MELD 数据集，包含训练、推理、可视化和多项优化方案说明。

## 🔍 项目简介

本仓库实现了基于图神经网络的多模态情感识别系统，主要特点包括：

- 多模态融合：音频、视觉、文本三模态融合
- 图谱优化：基于图谱谱域的图神经网络结构
- 对比学习：跨模态对齐与对比目标优化
- 可视化：训练曲线、指标分析与自动报告生成
- 支持 `IEMOCAP` 与 `MELD` 数据集

## 📁 目录结构

- `train_fourier.py`：主训练脚本，含多模态融合、图模型、优化损失和训练历史保存
- `inference.py`：训练后模型推理脚本
- `dataloader.py`：数据加载与预处理代码
- `FourierGNNmodel.py`：傅里叶图神经网络与图构建模型定义
- `models.py`：基础模型和辅助损失实现
- `augs.py`：数据增强/预处理工具
- `visualize_training.py`：交互式训练历史可视化
- `visualize_training_static.py`：静态 PNG 可视化生成
- `IEMOCAP_features/`：IEMOCAP 特征文件
- `MELD_features/`：MELD 特征文件
- `saved/`：模型权重与训练历史文件保存目录
- `*.md`：优化说明文档
  - `MODAL_OPTIMIZATION_README.md`
  - `CLASSIFIER_OPTIMIZATION_README.md`
  - `FOURIER_GNN_OPTIMIZATION.md`
  - `GRAPH_CONSTRUCTION_OPTIMIZATION.md`
  - `CONTRASTIVE_OPTIMIZATION.md`
  - `VISUALIZATION_GUIDE.md`

## ⚙️ 环境依赖

建议使用以下环境：

- Python 3.8+
- PyTorch 1.7+ 或更高
- torch-geometric 1.7+（如果使用图神经网络相关模块）
- matplotlib
- scikit-learn

安装示例：

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib scikit-learn numpy
```

## 🚀 训练指南

在项目根目录运行：

```bash
python train_fourier.py
```

### 常用参数示例

```bash
python train_fourier.py \
  --Dataset IEMOCAP \
  --epochs 250 \
  --batch-size 32 \
  --fusion_method concat \
  --modals avl \
  --graph_type relation \
  --graph_construct full \
  --loss_type cross_entropy \
  --use_speaker \
  --use_modal
```

### 主要命令行参数

- `--Dataset`：选择数据集，`IEMOCAP` 或 `MELD`
- `--epochs`：训练轮数
- `--batch-size`：批量大小
- `--fusion_method`：融合方式，`concat` / `gated` / `enhanced_fusion`
- `--modals`：模态组合，`a` / `v` / `l` / `av` / `al` / `vl` / `avl`
- `--graph_type`：图模型类型，`relation` / `GCN3` / `DeepGCN` / `MMGCN` / `MMGCN2`
- `--graph_construct`：图构建策略，`full` / `single` / `window` / `direct`
- `--loss_type`：损失类型，`cross_entropy` / `focal` / `label_smoothing`
- `--class-weight`：是否启用类别权重损失
- `--nodal-attention`：是否启用节点注意力
- `--use_speaker`：是否使用说话人嵌入
- `--use_modal`：是否使用模态嵌入

### 训练结果保存

训练过程中会自动保存：

- `saved/<DATASET>/<model_name>_best_model.pth`
- `saved/<DATASET>/<model_name>_history.json`

历史文件包含：
- `epochs`
- `train_loss`, `valid_loss`, `test_loss`
- `train_acc`, `valid_acc`, `test_acc`
- `train_fscore`, `valid_fscore`, `test_fscore`

## 🔎 推理指南

使用已保存的模型进行推理：

```bash
python inference.py --model saved/IEMOCAP/concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal_best_model.pth --sample 0
```

该脚本会加载模型并对指定测试样本输出情感概率。默认情感标签：`Angry`, `Excited`, `Frustrated`, `Happy`, `Neutral`, `Sad`。

## 📊 可视化指南

### 交互式可视化

```bash
python visualize_training.py --model-name concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal --save-dir ./saved/IEMOCAP/ --output-dir ./visualization/IEMOCAP/ --detail
```

### 静态图像可视化（推荐）

```bash
python visualize_training_static.py --model-name concat_avl_relation_fullusing_lstm_IEMOCAP_speaker_modal --save-dir ./saved/IEMOCAP/ --output-dir ./visualization/IEMOCAP/ --detail
```

生成结果包括：
- `training_overview.png`
- `loss_analysis.png`
- `accuracy_analysis.png`
- `fscore_analysis.png`

## 📘 文档与优化说明

仓库中包含以下优化说明文档，便于快速理解和扩展：

- `MODAL_OPTIMIZATION_README.md`：模态编码与融合优化
- `CLASSIFIER_OPTIMIZATION_README.md`：情感分类器优化
- `FOURIER_GNN_OPTIMIZATION.md`：傅里叶图神经网络优化
- `GRAPH_CONSTRUCTION_OPTIMIZATION.md`：图构建与边注意力优化
- `CONTRASTIVE_OPTIMIZATION.md`：对比学习优化
- `VISUALIZATION_GUIDE.md`：训练结果可视化指南

## 🧠 关键功能与改进

- `FourierGNNmodel.py` 中实现傅里叶谱域图神经网络与边注意力优化
- `train_fourier.py` 中支持多模态融合、图模型、类别平衡损失和训练历史保存
- `visualize_training_static.py` 支持无头环境静态可视化
- `inference.py` 支持快速模型推断
- 对比学习、模态融合和分类器部分均拥有专门优化文档

## ⚠️ 注意事项

- 请确保 `IEMOCAP_features/` 和 `MELD_features/` 中的数据已准备好
- 训练时建议使用 GPU
- 若存在大模型文件或历史文件，请通过 `.gitignore` 过滤不需要上传的临时文件

## ☁️ 部署与贡献

欢迎继续扩展本项目：

- 添加更多多模态数据集支持
- 引入更多图网络结构
- 完善可视化报告与实验跟踪
- 集成 TensorBoard 或 Weights & Biases

