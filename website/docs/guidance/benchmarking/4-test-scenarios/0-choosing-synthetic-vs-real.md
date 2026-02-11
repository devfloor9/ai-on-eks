---
sidebar_label: Choosing Between Synthetic and Real Data
---

# Choosing Between Synthetic and Real Dataset Testing

## Default Recommendation: Use Real Data Whenever Possible

Real production data provides the most accurate performance predictions because:

* Input token distribution matches actual user behavior
* Query complexity reflects real use cases
* Performance results directly correlate to production experience
* Identifies issues with specific prompt patterns that synthetic data might miss

## When to Use Synthetic Data (Scenarios 1-4):

* Initial deployment validation before production data exists
* Standardized comparisons across different systems (apples-to-apples)
* Testing extreme edge cases (very long prompts, burst patterns)
* Quick CI/CD validation where consistency matters more than realism
* Public benchmarking where real data cannot be shared

## Best Practice: Continuous Dataset Validation

Production workloads evolve over time. To ensure your benchmarks remain representative:


1. **Capture anonymized production prompts** periodically for benchmark datasets
2. **Monitor distribution drift** between test data and production traffic:

```bash
   # Compare token length distributions
   # Production: median=450, p95=1200
   # Test data: median=512, p95=2048
   # → Test data may overestimate TTFT
```

3. **Refresh test datasets quarterly** to match current production patterns
4. **Version your datasets** to track performance changes over time

This mirrors traditional ML continuous evaluation practices applied to inference performance testing.
