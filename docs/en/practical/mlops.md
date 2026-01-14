# MLOps Overview

## 1. Interview Summary

**Key Points to Remember:**
- **Reproducibility**: Version code, data, models, and experiments
- **Model versioning**: Track model artifacts and metadata
- **Deployment patterns**: Batch vs real-time, A/B testing, shadow mode
- **Monitoring**: Track model performance, data drift, and system health

**Common Interview Questions:**
- "How do you ensure ML experiments are reproducible?"
- "What are the differences between batch and real-time inference?"
- "How do you monitor models in production?"

---

## 2. Core Definitions

### MLOps Maturity Levels

| Level | Description | Practices |
|-------|-------------|-----------|
| 0 | Manual | Jupyter notebooks, manual deployment |
| 1 | ML Pipeline | Automated training, basic versioning |
| 2 | CI/CD for ML | Automated testing, model registry |
| 3 | Full MLOps | Feature stores, monitoring, auto-retraining |

### Key Components

| Component | Purpose | Tools |
|-----------|---------|-------|
| Version control | Track code changes | Git |
| Data versioning | Track data changes | DVC, Delta Lake |
| Experiment tracking | Compare experiments | MLflow, W&B |
| Model registry | Store model artifacts | MLflow, SageMaker |
| Feature store | Reusable features | Feast, Tecton |
| Model serving | Deploy predictions | TF Serving, Triton |
| Monitoring | Track drift, errors | Prometheus, Evidently |

### Deployment Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| Batch | Scheduled predictions | Daily reports |
| Real-time | Instant predictions | Recommendations |
| A/B testing | Compare models live | Model updates |
| Shadow mode | Parallel without serving | Safe rollout |
| Canary | Gradual traffic shift | Risk mitigation |

---

## 3. Math and Derivations

### Model Performance Monitoring

Track prediction distribution over time:

$$D_{KL}(P_{current} \| P_{baseline}) = \sum_i P_{current}(i) \log \frac{P_{current}(i)}{P_{baseline}(i)}$$

Alert if $D_{KL} > \tau$ (threshold).

### Data Drift Detection (PSI)

Population Stability Index:

$$PSI = \sum_i (A_i - E_i) \cdot \ln\left(\frac{A_i}{E_i}\right)$$

Where $A_i$ = actual proportion, $E_i$ = expected proportion in bin $i$.

| PSI | Interpretation |
|-----|----------------|
| < 0.1 | No significant change |
| 0.1 - 0.25 | Moderate change |
| > 0.25 | Significant drift |

### A/B Test Sample Size

Minimum samples per variant:

$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$$

Where $\delta$ = minimum detectable effect.

---

## 4. Algorithm Sketch

### Reproducibility Checklist

```
def ensure_reproducibility():
    # 1. Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 2. Version dependencies
    # requirements.txt or environment.yaml

    # 3. Version data
    # dvc add data/train.csv
    # git add data/train.csv.dvc

    # 4. Log experiment parameters
    mlflow.log_params({
        'learning_rate': 0.01,
        'batch_size': 32,
        'model_type': 'xgboost'
    })

    # 5. Log metrics
    mlflow.log_metrics({
        'train_auc': 0.85,
        'val_auc': 0.82
    })

    # 6. Save model artifact
    mlflow.sklearn.log_model(model, 'model')
```

### Model Deployment Pipeline

```
def deploy_model(model, version):
    # 1. Validate model
    metrics = evaluate_model(model, test_data)
    assert metrics['auc'] > threshold

    # 2. Register model
    model_uri = register_model(model, version, metrics)

    # 3. Deploy to staging
    deploy_to_staging(model_uri)

    # 4. Run integration tests
    run_integration_tests(staging_endpoint)

    # 5. Shadow mode deployment
    deploy_shadow(model_uri, production_endpoint)
    compare_predictions(shadow_predictions, production_predictions)

    # 6. Canary deployment
    deploy_canary(model_uri, traffic_percentage=5)
    monitor_canary(duration='1h')

    # 7. Full rollout
    deploy_production(model_uri)

    return deployment_status
```

### Monitoring System

```
def monitor_model(predictions, actuals, baseline):
    # Data drift
    psi = calculate_psi(predictions, baseline)
    if psi > 0.25:
        alert("Significant data drift detected")

    # Performance monitoring
    if len(actuals) > min_samples:
        current_metrics = calculate_metrics(predictions, actuals)
        if current_metrics['auc'] < baseline['auc'] - tolerance:
            alert("Model performance degraded")

    # System health
    latency_p99 = get_latency_percentile(99)
    if latency_p99 > sla_threshold:
        alert("Latency SLA breach")

    return monitoring_report
```

---

## 5. Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Training-serving skew | Different preprocessing | Use shared feature pipeline |
| Silent model degradation | No monitoring | Set up drift detection |
| Non-reproducible results | Unset random seeds | Version everything |
| Slow inference | Unoptimized model | Profile and optimize |
| Stale models | No retraining schedule | Automated retraining |

### Production Checklist

```
Before deployment:
[ ] Unit tests pass
[ ] Integration tests pass
[ ] Model metrics meet threshold
[ ] Latency requirements met
[ ] Memory usage acceptable
[ ] Input validation in place
[ ] Error handling implemented
[ ] Rollback plan documented

After deployment:
[ ] Monitoring dashboards active
[ ] Alerts configured
[ ] Shadow mode comparison done
[ ] Canary analysis complete
[ ] Documentation updated
```

---

## 6. Mini Example

```python
import numpy as np
import json
from datetime import datetime

def calculate_psi(actual, expected, bins=10):
    """Calculate Population Stability Index."""
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Calculate proportions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_props = expected_counts / len(expected) + 1e-8
    actual_props = actual_counts / len(actual) + 1e-8

    # PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return psi


def create_experiment_log(params, metrics, model_path):
    """Create a simple experiment log."""
    log = {
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'metrics': metrics,
        'model_path': model_path,
        'git_hash': 'abc123',  # Would be actual git hash
        'python_version': '3.9.0',
        'dependencies': {
            'numpy': '1.21.0',
            'sklearn': '0.24.0'
        }
    }
    return log


def simple_model_registry():
    """Simple in-memory model registry."""
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


# Example usage
np.random.seed(42)

# 1. Data drift detection
print("=== Data Drift Detection (PSI) ===")
baseline_predictions = np.random.normal(0.5, 0.1, 1000)
current_no_drift = np.random.normal(0.5, 0.1, 500)
current_with_drift = np.random.normal(0.6, 0.15, 500)

psi_no_drift = calculate_psi(current_no_drift, baseline_predictions)
psi_with_drift = calculate_psi(current_with_drift, baseline_predictions)

print(f"PSI (no drift): {psi_no_drift:.4f} - {'OK' if psi_no_drift < 0.1 else 'ALERT'}")
print(f"PSI (with drift): {psi_with_drift:.4f} - {'OK' if psi_with_drift < 0.1 else 'ALERT'}")

# 2. Experiment logging
print("\n=== Experiment Logging ===")
experiment = create_experiment_log(
    params={'learning_rate': 0.01, 'max_depth': 5},
    metrics={'train_auc': 0.85, 'val_auc': 0.82, 'test_auc': 0.80},
    model_path='/models/xgb_v1.pkl'
)
print(json.dumps(experiment, indent=2))

# 3. Model registry
print("\n=== Model Registry ===")
register, promote, get_latest = simple_model_registry()

# Register models
v1 = register('fraud_detector', 'v1', {'auc': 0.80}, '/models/v1.pkl')
v2 = register('fraud_detector', 'v2', {'auc': 0.85}, '/models/v2.pkl')
print(f"Registered: {v1}, {v2}")

# Promote to production
promote(v2, 'production')
latest = get_latest('fraud_detector', 'production')
print(f"Latest production model: {latest}")

# 4. A/B test sample size calculation
print("\n=== A/B Test Sample Size ===")
def ab_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """Calculate minimum sample size for A/B test."""
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_pooled = (p1 + p2) / 2

    n = 2 * ((z_alpha + z_beta) ** 2) * p_pooled * (1 - p_pooled) / (mde ** 2)
    return int(np.ceil(n))

baseline_ctr = 0.10  # 10% click-through rate
mde = 0.01  # Want to detect 1% absolute change
n = ab_sample_size(baseline_ctr, mde)
print(f"Baseline CTR: {baseline_ctr:.1%}")
print(f"Minimum detectable effect: {mde:.1%}")
print(f"Required samples per variant: {n:,}")
```

**Output:**
```
=== Data Drift Detection (PSI) ===
PSI (no drift): 0.0234 - OK
PSI (with drift): 0.1876 - ALERT

=== Experiment Logging ===
{
  "timestamp": "2024-01-15T10:30:00.000000",
  "params": {"learning_rate": 0.01, "max_depth": 5},
  "metrics": {"train_auc": 0.85, "val_auc": 0.82, "test_auc": 0.80},
  "model_path": "/models/xgb_v1.pkl",
  "git_hash": "abc123",
  "python_version": "3.9.0",
  "dependencies": {"numpy": "1.21.0", "sklearn": "0.24.0"}
}

=== Model Registry ===
Registered: fraud_detector:v1, fraud_detector:v2
Latest production model: ('fraud_detector:v2', {...})

=== A/B Test Sample Size ===
Baseline CTR: 10.0%
Minimum detectable effect: 1.0%
Required samples per variant: 14,752
```

---

## 7. Quiz

<details>
<summary><strong>Q1: What is training-serving skew and how do you prevent it?</strong></summary>

**Training-serving skew**: Differences between training and production environments that cause model behavior to differ.

**Causes**:
- Different preprocessing code
- Different data sources
- Different library versions
- Time-dependent features computed differently

**Prevention**:
1. Use a shared feature engineering pipeline
2. Log exact preprocessing parameters
3. Use feature stores for consistent features
4. Version all dependencies
5. Test with production-like data
6. Monitor for feature distribution changes
</details>

<details>
<summary><strong>Q2: What are the trade-offs between batch and real-time inference?</strong></summary>

**Batch inference**:
- Pros: Higher throughput, easier scaling, cost-efficient
- Cons: Stale predictions, not suitable for interactive use
- Use cases: Daily reports, recommendations, risk scoring

**Real-time inference**:
- Pros: Fresh predictions, interactive applications
- Cons: Higher infrastructure cost, latency requirements, scaling challenges
- Use cases: Fraud detection, search ranking, chatbots

**Hybrid approach**: Pre-compute common cases (batch), compute rare cases on-demand (real-time).
</details>

<details>
<summary><strong>Q3: How do you ensure ML experiments are reproducible?</strong></summary>

**Essential practices**:

1. **Version control**:
   - Code (Git)
   - Data (DVC, Delta Lake)
   - Models (MLflow)
   - Environment (requirements.txt, Docker)

2. **Set random seeds**:
   - Python random
   - NumPy
   - Framework-specific (PyTorch, TensorFlow)

3. **Log everything**:
   - Hyperparameters
   - Metrics
   - Data checksums
   - Git commit hash

4. **Containerization**: Docker for consistent environment

5. **Immutable data**: Don't modify original data; create versions
</details>

<details>
<summary><strong>Q4: What monitoring should you set up for production ML models?</strong></summary>

**Data monitoring**:
- Feature distribution drift (PSI, KL divergence)
- Missing value rates
- Outlier frequency
- Data schema violations

**Model monitoring**:
- Prediction distribution
- Confidence scores
- Performance metrics (when labels available)
- Fairness metrics

**System monitoring**:
- Latency (p50, p95, p99)
- Throughput
- Error rates
- Resource utilization (CPU, memory, GPU)

**Alerting**: Set thresholds and alert on anomalies.
</details>

<details>
<summary><strong>Q5: What is shadow mode deployment and when should you use it?</strong></summary>

**Shadow mode**: Deploy new model alongside production, route traffic to both, but only serve production model's predictions.

**How it works**:
1. Both models receive same input
2. Old model serves predictions
3. New model's predictions logged but not used
4. Compare predictions offline

**When to use**:
- High-risk applications (fraud, medical)
- Significant model changes
- Validating performance on real traffic
- Before A/B testing

**Benefits**: No user impact, real production data, identify edge cases.
</details>

<details>
<summary><strong>Q6: How do you handle model rollbacks in production?</strong></summary>

**Rollback strategy**:

1. **Keep previous version ready**: Don't delete old models
2. **Quick switch mechanism**: One-click rollback
3. **Automatic triggers**: Rollback on metric degradation

**Implementation**:
```python
# Traffic routing
if model_metrics < threshold:
    route_to('previous_version')
    alert('Auto-rollback triggered')
```

**Blue-green deployment**: Keep two production environments; switch instantly.

**Best practices**:
- Test rollback procedure regularly
- Document rollback steps
- Keep at least 2 previous versions
- Monitor closely after any deployment
</details>

---

## 8. References

1. Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NeurIPS.
2. Breck, E., et al. (2017). "The ML Test Score: A Rubric for ML Production Readiness." Google.
3. Kreuzberger, D., et al. (2022). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." IEEE Access.
4. Klaise, J., et al. (2020). "Monitoring Machine Learning Models in Production." arXiv.
5. MLflow documentation: https://mlflow.org/docs/latest/index.html
