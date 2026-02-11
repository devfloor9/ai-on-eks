apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: ${name}
spec:
  ${indent(2, ec2nodeclass)}
