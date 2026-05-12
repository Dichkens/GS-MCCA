# 对比学习优化

## 1. 优化前现状

项目中当前只有 `models.py` 中的一个对比学习模块：`CPC`。
原始实现通过从一个模态预测另一个模态，并将正样本的内积与所有预测的 log-sum-exp 结合来计算损失。

存在问题：
- 损失函数设计将正负项混合在一起，形式不够标准。
- 未明确采用温度缩放的 InfoNCE 目标。
- 缺少用于稳定表示对齐的专用投影头。
- 没有对称的跨模态对比损失，难以保证多模态表示的一致性。

## 2. 优化目标

- 通过归一化嵌入和稳定的 InfoNCE 来增强对比学习稳定性。
- 引入可复用的投影头，改善跨模态表示学习。
- 新增对称的跨模态对比损失，强化配对模态对齐能力。
- 保留原始 `CPC` 思路，同时将评分公式替换为现代对比学习形式。

## 3. 在 `models.py` 中实现的改动

### 3.1 `ProjectionHead`

新增 `ProjectionHead`，一个轻量 MLP，可选批归一化、ReLU 和 dropout。
它将原始嵌入映射到归一化的投影空间，在该空间中计算对比相似度。

### 3.2 `NTXentLoss`

新增归一化温度缩放交叉熵损失。
该损失对两个归一化嵌入集合计算相似度 logits，并双向应用交叉熵。
这是对比学习中的标准目标，有助于避免表示塌缩并利用批内负样本。

### 3.3 `CrossModalContrastive`

新增对称跨模态对比模块，用于多模态特征对齐。
它对两种模态进行投影，然后使用 `NTXentLoss`，使模态表示在共享空间中对齐。

### 3.4 改进 `CPC`

将原始 `CPC` 模块更新为稳定的 InfoNCE 版本：
- 从模态 `y` 预测模态 `x`
- 对预测向量和真实向量都做归一化
- 应用温度缩放 logits
- 通过批内负样本计算交叉熵损失

## 4. 新对比组件的使用方式

训练示例：

```python
from models import CrossModalContrastive, CPC

# 两模态嵌入的跨模态对比损失
contrastive = CrossModalContrastive(x_size=512, y_size=256, projection_dim=128, temperature=0.07)
loss, z_x, z_y = contrastive(x_repr, y_repr)

# 原始 CPC 风格的 InfoNCE 损失
cpc_loss = CPC(x_size=512, y_size=256, n_layers=2, activation='Tanh', temperature=0.07)
loss = cpc_loss(x_repr, y_repr)
```

## 5. 优化收益

- 通过归一化和温度缩放提升数值稳定性。
- 明确的跨模态对比目标，适合多模态情感识别场景。
- 更好的模块化设计，将投影和损失计算分离。
- 更容易与图结构或融合模型集成：先投影再计算相似度。

## 6. 建议后续步骤

1. 将 `CrossModalContrastive` 集成到训练流水线中，用于模态对齐。
2. 与分类损失联合训练，例如 `total_loss = ce_loss + alpha * contrastive_loss`。
3. 通过调整投影维度和温度参数，调节对齐强度。
4. 如果使用超过两种模态，可扩展为多模态成对组合或多视角对比头。
