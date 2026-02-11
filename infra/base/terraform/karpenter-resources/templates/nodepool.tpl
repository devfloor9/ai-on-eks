apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: ${name}
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 300s
    consolidationPolicy: WhenEmpty
  template:
    metadata:
      labels:
        amiFamily: ${ami_family}
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: ${name}
%{ if taints != "" ~}
      taints:
        - key: ${taints}
          effect: "NoSchedule"
%{ endif ~}
      requirements:
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values:
            - ${instance_family}
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
            - spot
      terminationGracePeriod: 48h
  weight: 100
