# Multi-Model Routing

Route requests to different AI models based on the `x-ai-eg-model` header.

## Prerequisites

Deploy common infrastructure first:
```bash
cd ../
kubectl apply -f gateway.yaml
```

## Deploy

```bash
kubectl apply -f multi-model-routing/ai-gateway-route.yaml
```

## Test

```bash
python3 multi-model-routing/client.py
```

## Manual Testing

```bash
# Grab the Gateway URL
export GATEWAY_URL=`kubectl get gateway ai-gateway -o jsonpath={.status.addresses[0].value}`

# Test gpt-oss-20b
curl -X POST http://$GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-ai-eg-model: openai/gpt-oss-20b" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'

curl -X POST http://$GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-ai-eg-model: Qwen/Qwen3-1.7B" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Troubleshooting

- **404 errors**: Check if `ai-gateway-route.yaml` is applied
- **Model not responding**: Verify backend services are running
- **Model invoke timing out**: Verify if envoy default AI gateway deployment has scaled up and running at least 1 pod. If not, restart the envoy AI gateway deployment.
```bash
kubectl rollout restart deployment ai-gateway-controller -n envoy-ai-gateway-system
```

```bash
# Check route status
kubectl get aigatewayroute multi-model-route -o yaml

# Check backend services
kubectl get svc | grep -E "(qwen|gpt-oss)"
```
