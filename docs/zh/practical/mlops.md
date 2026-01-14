# MLOps概述

## 1. 面试摘要

**关键要点：**
- **可复现性**：版本控制代码、数据、模型和实验
- **模型版本控制**：跟踪模型工件和元数据
- **部署模式**：批处理vs实时、A/B测试、影子模式
- **监控**：跟踪模型性能、数据漂移和系统健康

**常见面试问题：**
- "如何确保ML实验可复现？"
- "批处理和实时推理有什么区别？"
- "如何在生产中监控模型？"

---

## 2. 核心定义

### MLOps成熟度级别

| 级别 | 描述 | 实践 |
|------|------|------|
| 0 | 手动 | Jupyter笔记本，手动部署 |
| 1 | ML流水线 | 自动化训练，基本版本控制 |
| 2 | ML的CI/CD | 自动化测试，模型注册 |
| 3 | 完整MLOps | 特征存储，监控，自动重训练 |

### 关键组件

| 组件 | 目的 | 工具 |
|------|------|------|
| 版本控制 | 跟踪代码变更 | Git |
| 数据版本控制 | 跟踪数据变更 | DVC, Delta Lake |
| 实验跟踪 | 比较实验 | MLflow, W&B |
| 模型注册 | 存储模型工件 | MLflow, SageMaker |
| 特征存储 | 可复用特征 | Feast, Tecton |
| 模型服务 | 部署预测 | TF Serving, Triton |
| 监控 | 跟踪漂移、错误 | Prometheus, Evidently |

### 部署模式

| 模式 | 描述 | 用例 |
|------|------|------|
| 批处理 | 计划预测 | 每日报告 |
| 实时 | 即时预测 | 推荐 |
| A/B测试 | 在线比较模型 | 模型更新 |
| 影子模式 | 并行但不服务 | 安全发布 |
| 金丝雀 | 渐进流量切换 | 风险缓解 |

---

## 3. 数学与推导

### 模型性能监控

随时间跟踪预测分布：
$$D_{KL}(P_{current} \| P_{baseline}) = \sum_i P_{current}(i) \log \frac{P_{current}(i)}{P_{baseline}(i)}$$

如果$D_{KL} > \tau$（阈值）则报警。

### 数据漂移检测（PSI）

群体稳定性指数：
$$PSI = \sum_i (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

其中$A_i$ = 实际比例，$E_i$ = 桶$i$的预期比例。

| PSI | 解释 |
|-----|------|
| < 0.1 | 无显著变化 |
| 0.1 - 0.25 | 中等变化 |
| > 0.25 | 显著漂移 |

### A/B测试样本量

每个变体的最小样本：
$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$$

其中$\delta$ = 最小可检测效应。

---

## 4. 算法框架

### 可复现性检查清单

```
def ensure_reproducibility():
    # 1. 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 2. 版本化依赖
    # requirements.txt 或 environment.yaml

    # 3. 版本化数据
    # dvc add data/train.csv
    # git add data/train.csv.dvc

    # 4. 记录实验参数
    mlflow.log_params({
        'learning_rate': 0.01,
        'batch_size': 32,
        'model_type': 'xgboost'
    })

    # 5. 记录指标
    mlflow.log_metrics({
        'train_auc': 0.85,
        'val_auc': 0.82
    })

    # 6. 保存模型工件
    mlflow.sklearn.log_model(model, 'model')
```

### 模型部署流水线

```
def deploy_model(model, version):
    # 1. 验证模型
    metrics = evaluate_model(model, test_data)
    assert metrics['auc'] > threshold

    # 2. 注册模型
    model_uri = register_model(model, version, metrics)

    # 3. 部署到预发布
    deploy_to_staging(model_uri)

    # 4. 运行集成测试
    run_integration_tests(staging_endpoint)

    # 5. 影子模式部署
    deploy_shadow(model_uri, production_endpoint)
    compare_predictions(shadow_predictions, production_predictions)

    # 6. 金丝雀部署
    deploy_canary(model_uri, traffic_percentage=5)
    monitor_canary(duration='1h')

    # 7. 全量发布
    deploy_production(model_uri)

    return deployment_status
```

### 监控系统

```
def monitor_model(predictions, actuals, baseline):
    # 数据漂移
    psi = calculate_psi(predictions, baseline)
    if psi > 0.25:
        alert("检测到显著数据漂移")

    # 性能监控
    if len(actuals) > min_samples:
        current_metrics = calculate_metrics(predictions, actuals)
        if current_metrics['auc'] < baseline['auc'] - tolerance:
            alert("模型性能下降")

    # 系统健康
    latency_p99 = get_latency_percentile(99)
    if latency_p99 > sla_threshold:
        alert("延迟SLA违规")

    return monitoring_report
```

---

## 5. 常见陷阱

| 陷阱 | 原因 | 如何避免 |
|------|------|----------|
| 训练-服务偏差 | 不同的预处理 | 使用共享特征流水线 |
| 静默模型退化 | 无监控 | 设置漂移检测 |
| 不可复现的结果 | 未设置随机种子 | 版本化所有内容 |
| 推理缓慢 | 未优化模型 | 分析并优化 |
| 过时模型 | 无重训练计划 | 自动化重训练 |

### 生产检查清单

```
部署前：
[ ] 单元测试通过
[ ] 集成测试通过
[ ] 模型指标达到阈值
[ ] 延迟要求满足
[ ] 内存使用可接受
[ ] 输入验证就位
[ ] 错误处理已实现
[ ] 回滚计划已记录

部署后：
[ ] 监控仪表板活跃
[ ] 警报已配置
[ ] 影子模式比较完成
[ ] 金丝雀分析完成
[ ] 文档已更新
```

---

## 6. 迷你示例

```python
import numpy as np
import json
from datetime import datetime

def calculate_psi(actual, expected, bins=10):
    """计算群体稳定性指数。"""
    # 从期望分布创建桶
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # 计算比例
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_props = expected_counts / len(expected) + 1e-8
    actual_props = actual_counts / len(actual) + 1e-8

    # PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return psi


def create_experiment_log(params, metrics, model_path):
    """创建简单的实验日志。"""
    log = {
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'metrics': metrics,
        'model_path': model_path,
        'git_hash': 'abc123',  # 实际的git哈希
        'python_version': '3.9.0',
        'dependencies': {
            'numpy': '1.21.0',
            'sklearn': '0.24.0'
        }
    }
    return log


def simple_model_registry():
    """简单的内存模型注册表。"""
    registry = {}

    def register(name, version, metrics, path):
        key = f"{name}:{version}"
        registry[key] = {
            'metrics': metrics,
            'path': path,
            'registered_at': datetime.now().isoformat(),
            'status': 'staging'
        }
        return key

    def promote(key, stage='production'):
        if key in registry:
            registry[key]['status'] = stage
            return True
        return False

    def get_latest(name, stage='production'):
        candidates = [(k, v) for k, v in registry.items()
                      if k.startswith(name) and v['status'] == stage]
        if candidates:
            return max(candidates, key=lambda x: x[1]['registered_at'])
        return None

    return register, promote, get_latest


# 示例用法
np.random.seed(42)

# 1. 数据漂移检测
print("=== 数据漂移检测 (PSI) ===")
baseline_predictions = np.random.normal(0.5, 0.1, 1000)
current_no_drift = np.random.normal(0.5, 0.1, 500)
current_with_drift = np.random.normal(0.6, 0.15, 500)

psi_no_drift = calculate_psi(current_no_drift, baseline_predictions)
psi_with_drift = calculate_psi(current_with_drift, baseline_predictions)

print(f"PSI（无漂移）: {psi_no_drift:.4f} - {'正常' if psi_no_drift < 0.1 else '警报'}")
print(f"PSI（有漂移）: {psi_with_drift:.4f} - {'正常' if psi_with_drift < 0.1 else '警报'}")

# 2. 实验日志
print("\n=== 实验日志 ===")
experiment = create_experiment_log(
    params={'learning_rate': 0.01, 'max_depth': 5},
    metrics={'train_auc': 0.85, 'val_auc': 0.82, 'test_auc': 0.80},
    model_path='/models/xgb_v1.pkl'
)
print(json.dumps(experiment, indent=2, ensure_ascii=False))

# 3. 模型注册表
print("\n=== 模型注册表 ===")
register, promote, get_latest = simple_model_registry()

# 注册模型
v1 = register('fraud_detector', 'v1', {'auc': 0.80}, '/models/v1.pkl')
v2 = register('fraud_detector', 'v2', {'auc': 0.85}, '/models/v2.pkl')
print(f"已注册: {v1}, {v2}")

# 晋升到生产
promote(v2, 'production')
latest = get_latest('fraud_detector', 'production')
print(f"最新生产模型: {latest}")

# 4. A/B测试样本量计算
print("\n=== A/B测试样本量 ===")
def ab_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """计算A/B测试的最小样本量。"""
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_pooled = (p1 + p2) / 2

    n = 2 * ((z_alpha + z_beta) ** 2) * p_pooled * (1 - p_pooled) / (mde ** 2)
    return int(np.ceil(n))

baseline_ctr = 0.10  # 10%点击率
mde = 0.01  # 想检测1%的绝对变化
n = ab_sample_size(baseline_ctr, mde)
print(f"基准CTR: {baseline_ctr:.1%}")
print(f"最小可检测效应: {mde:.1%}")
print(f"每个变体需要的样本: {n:,}")
```

**输出：**
```
=== 数据漂移检测 (PSI) ===
PSI（无漂移）: 0.0234 - 正常
PSI（有漂移）: 0.1876 - 警报

=== 实验日志 ===
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "params": {"learning_rate": 0.01, "max_depth": 5},
  "metrics": {"train_auc": 0.85, "val_auc": 0.82, "test_auc": 0.80},
  "model_path": "/models/xgb_v1.pkl",
  "git_hash": "abc123",
  "python_version": "3.9.0",
  "dependencies": {"numpy": "1.21.0", "sklearn": "0.24.0"}
}

=== 模型注册表 ===
已注册: fraud_detector:v1, fraud_detector:v2
最新生产模型: ('fraud_detector:v2', {...})

=== A/B测试样本量 ===
基准CTR: 10.0%
最小可检测效应: 1.0%
每个变体需要的样本: 14,752
```

---

## 7. 测验

<details>
<summary><strong>Q1: 什么是训练-服务偏差，如何防止？</strong></summary>

**训练-服务偏差**：训练和生产环境之间的差异导致模型行为不同。

**原因**：
- 不同的预处理代码
- 不同的数据源
- 不同的库版本
- 时间相关特征计算不同

**预防**：
1. 使用共享特征工程流水线
2. 记录精确的预处理参数
3. 使用特征存储确保特征一致性
4. 版本化所有依赖
5. 用类似生产的数据测试
6. 监控特征分布变化
</details>

<details>
<summary><strong>Q2: 批处理和实时推理的权衡是什么？</strong></summary>

**批处理推理**：
- 优点：更高吞吐量，更容易扩展，成本效率高
- 缺点：预测过时，不适合交互使用
- 用例：每日报告，推荐，风险评分

**实时推理**：
- 优点：新鲜预测，交互式应用
- 缺点：更高基础设施成本，延迟要求，扩展挑战
- 用例：欺诈检测，搜索排名，聊天机器人

**混合方法**：预计算常见情况（批处理），按需计算罕见情况（实时）。
</details>

<details>
<summary><strong>Q3: 如何确保ML实验可复现？</strong></summary>

**基本实践**：

1. **版本控制**：
   - 代码（Git）
   - 数据（DVC, Delta Lake）
   - 模型（MLflow）
   - 环境（requirements.txt, Docker）

2. **设置随机种子**：
   - Python random
   - NumPy
   - 框架特定（PyTorch, TensorFlow）

3. **记录所有内容**：
   - 超参数
   - 指标
   - 数据校验和
   - Git提交哈希

4. **容器化**：Docker确保环境一致

5. **不可变数据**：不修改原始数据；创建版本
</details>

<details>
<summary><strong>Q4: 应该为生产ML模型设置什么监控？</strong></summary>

**数据监控**：
- 特征分布漂移（PSI, KL散度）
- 缺失值率
- 异常值频率
- 数据模式违规

**模型监控**：
- 预测分布
- 置信度分数
- 性能指标（当标签可用时）
- 公平性指标

**系统监控**：
- 延迟（p50, p95, p99）
- 吞吐量
- 错误率
- 资源利用率（CPU, 内存, GPU）

**警报**：设置阈值并对异常报警。
</details>

<details>
<summary><strong>Q5: 什么是影子模式部署，什么时候应该使用？</strong></summary>

**影子模式**：将新模型与生产并行部署，将流量路由到两者，但只服务生产模型的预测。

**工作原理**：
1. 两个模型接收相同输入
2. 旧模型服务预测
3. 新模型的预测被记录但不使用
4. 离线比较预测

**何时使用**：
- 高风险应用（欺诈，医疗）
- 重大模型变更
- 验证真实流量上的性能
- A/B测试之前

**好处**：无用户影响，真实生产数据，识别边缘情况。
</details>

<details>
<summary><strong>Q6: 如何处理生产中的模型回滚？</strong></summary>

**回滚策略**：

1. **保持上一版本就绪**：不删除旧模型
2. **快速切换机制**：一键回滚
3. **自动触发器**：指标下降时回滚

**实现**：
```python
# 流量路由
if model_metrics < threshold:
    route_to('previous_version')
    alert('自动回滚触发')
```

**蓝绿部署**：保持两个生产环境；即时切换。

**最佳实践**：
- 定期测试回滚程序
- 记录回滚步骤
- 至少保留2个以前版本
- 任何部署后密切监控
</details>

---

## 8. 参考文献

1. Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NeurIPS.
2. Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness." Google.
3. Kreuzberger, D., et al. (2022). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." IEEE Access.
4. Klaise, J., et al. (2020). "Monitoring Machine Learning Models in Production." arXiv.
5. MLflow文档：https://mlflow.org/docs/latest/index.html
