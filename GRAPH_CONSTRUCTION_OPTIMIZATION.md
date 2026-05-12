# 图构建优化说明

## 目标

对项目中的图构建部分进行优化，使图结构生成更高效、更稳定，并修复原来批量图组合中存在的节点拼接与掩码不一致问题。

## 优化内容

### 1. 边生成逻辑优化

- 原始实现使用 `set`、`union` 进行边对合并，存在重复计算和运行开销。
- 新实现直接按窗口范围顺序生成边对：
  - `window_past=-1` 表示当前节点之前所有节点都可连接
  - `window_future=-1` 表示当前节点之后所有节点都可连接
  - 否则仅连接过去/未来窗口以内的节点

**优势**:
- 计算时间更少
- 更容易保持边顺序一致
- 能避免集合操作带来的额外内存开销

### 2. 边注意力掩码稳定性增强

- 将 `MaskedEdgeAttention` 的设备管理统一为 `M.device`，避免手动区分 `cuda` 分支。
- 使用 `torch.zeros_like(alpha, device=device)` 统一创建掩码。
- 采用 `clamp_min(1e-9)` 保证求和不为 0，避免数值不稳定或 `NaN`。
- 如果当前批次没有边，直接返回与 `alpha` 同形状的零矩阵。

**优势**:
- 兼容性更强
- 数值稳定性更好
- 避免处理空图时的异常

### 3. 批量图构建改进

- `batch_graphify` 现在先统一生成所有批次边对，再调用 `att_model`。
- `features` 被正确转换为 `[batch, seq_len, dim]`，避免原始 `features[:, j, :]` 带来的顺序错误。
- 使用 `features[j, :seq_len, :]` 精确提取当前会话节点特征。
- `speaker_ids` 使用 `torch.argmax(qmask[:seq_len, j, :], dim=-1)` 一次性提取，避免多次查询。
- 统一使用当前 `device` 创建 `edge_index`, `edge_norm`, `edge_type`。
- 支持无边情况，返回空张量而不是失败。

**优势**:
- 图构建结果与输入序列对齐
- 明确批次边索引顺序
- 兼容不同长度对齐和空图情况

## 具体代码位置

- `FourierGNNmodel.py`
  - `edge_perms`
  - `MaskedEdgeAttention.forward`
  - `batch_graphify`

## 效果与验证

- 保持 `FourierGNNmodel.py` 语法检查通过
- 图构建逻辑更加可读、可维护
- 更少重复计算、稳定性提高

## 推荐使用方式

- 若要调整图构建范围，可直接修改 `window_past` 与 `window_future` 参数。
- 如需快速排查图结构，可在 `batch_graphify` 中打印 `edge_index_lengths` 或 `edge_type`。

## 未来可扩展方向

- 支持基于注意力门控的边稀疏化策略
- 加入说话人关系权重正则化
- 采用图采样机制减少长序列边数
- 将 `edge_norm` 归一化为 `edge_weight` 以适配更多图卷积库
