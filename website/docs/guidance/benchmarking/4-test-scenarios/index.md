---
sidebar_label: Test Scenarios
---

# Test Scenarios

This section provides practical test scenarios for benchmarking LLM inference performance. Each scenario addresses specific testing objectives and use cases.

## Available Scenarios

### [Choosing Between Synthetic and Real Dataset Testing](./0-choosing-synthetic-vs-real.md)
Understand when to use synthetic vs. real-world data for benchmarking and best practices for dataset selection.

### [Scenario 1: Baseline Performance](./1-baseline-performance.md)
Establish your system's optimal performance with zero contention. Ideal for understanding the best-case performance without queueing or resource competition.

**Use when:**
- Just deployed a new endpoint
- Made infrastructure changes
- Need a clean reference point for optimization

### [Scenario 2: Saturation Testing](./2-saturation-testing.md)
Determine maximum sustainable throughput before performance degrades through multi-stage load testing.

**Use when:**
- Planning capacity
- Setting autoscaling thresholds
- Validating before production launch

### [Scenario 3: Automatic Saturation Detection](./3-automatic-saturation-detection.md)
Use sweep mode for automated capacity discovery without manual QPS guessing.

**Use when:**
- Initial deployments
- CI/CD pipelines
- Quick capacity re-validation

### [Scenario 4: Production Simulation](./4-production-simulation.md)
Replicate real-world traffic with variable request sizes and bursty (Poisson) arrivals.

**Use when:**
- Final validation before launch
- Setting SLA targets
- Validating realistic workload handling

### [Scenario 5: Real Dataset Testing](./5-real-dataset-testing.md)
Validate production-ready performance using actual user prompts and query patterns.

**Use when:**
- Model fine-tuned for specific patterns
- Comparing model versions
- Need authentic performance guarantees

## Prerequisites

All scenarios use the [AI on EKS Benchmark Helm Chart](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/benchmark-charts) for deployment. Before proceeding:

1. **Install Helm** (version 3.x or later)
2. **Add the AI on EKS Helm repository:**
   ```bash
   helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
   helm repo update
   ```
3. **Configure kubectl** access to your EKS cluster
4. **Deploy your inference service** (e.g., vLLM serving your model)

## Implementation Notes

Each scenario below demonstrates deployment using the **Helm chart** as the recommended method. The chart provides:

- **Consistent configuration** across all test scenarios
- **Values-driven customization** for specific use cases
- **Production-ready defaults** with pod affinity and resource management
- **Easy maintenance** with centralized configuration

For educational purposes or custom deployments, each scenario also includes a collapsible section with raw Kubernetes YAML showing the complete manifest structure. This alternative approach uses **runtime dependency installation** where dependencies are installed in the main container at startup.
