# AWS Setup for Neuron CI

To run the neuron-marked tests (`pytest -m neuron`) on real Trainium hardware via GitHub Actions, you need:

1. An AWS account with access to trn1/trn2/inf2 instances
2. A Terraform-provisioned CI instance (see `infra/terraform/`)
3. GitHub repo secrets + variables configured

## One-time setup

### 1. Deploy the Terraform module

Pick a VPC and subnet. A private subnet with NAT egress is fine (SSM works without public IP). A public subnet also works.

```bash
cd infra/terraform

terraform init

terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx" \
  -var="instance_type=trn1.2xlarge" \
  -var="instance_tag=trnfft-ci-trn1"
```

Capture the outputs:

```
instance_id   = "i-0abc..."
instance_tag  = "trnfft-ci-trn1"
aws_role_arn  = "arn:aws:iam::123456789012:role/trnfft-ci-trn1-gh-actions"
aws_region    = "us-east-1"
```

### 2. Configure GitHub Actions

In the repo settings (`Settings → Secrets and variables → Actions`):

- **Secret** `AWS_ROLE_ARN` — value from Terraform output
- **Variable** `AWS_REGION` — value from Terraform output

### 3. Verify the instance is ready

Wait ~5 minutes after `terraform apply` for user-data to finish (clones trnfft, installs deps). Check SSM connectivity:

```bash
aws ssm describe-instance-information \
  --filters "Key=tag:Name,Values=trnfft-ci-trn1"
```

The instance should appear with `PingStatus: Online`. Then stop it — the workflow will start it on demand:

```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
```

## Running the workflow

Via GitHub CLI:

```bash
gh workflow run neuron.yml -R scttfrdmn/trnfft -f instance_type=trn1
```

Or the Actions tab → "Neuron Hardware Tests" → "Run workflow".

The workflow will:

1. Start the tagged instance
2. Wait for it to be running and SSM-addressable
3. Run `pytest tests/ -v -m neuron` via `aws ssm send-command`
4. Collect stdout/stderr into the job log
5. Stop the instance (even on failure)

## Cost

Stopped instances only cost EBS storage (~$10/mo for 100 GB gp3).

When running, on-demand pricing (us-east-1 as of 2026):

| Type | Hourly | Typical run (10 min) |
|------|-------:|---------------------:|
| trn1.2xlarge | $1.34 | $0.22 |
| trn2.8xlarge | $10.00 | $1.67 |
| inf2.xlarge | $0.76 | $0.13 |

## Troubleshooting

**Workflow fails at "Start instance" with "No instance found"**
— The tag doesn't match. Check `aws ec2 describe-instances --filters "Name=tag:Name,Values=trnfft-ci-trn1"`.

**Workflow fails at "Run neuron tests" with SSM error**
— Instance may not have SSM agent running. Check `aws ssm describe-instance-information`. User-data may still be running if you triggered the workflow too soon after `terraform apply`.

**Tests fail with "neuronxcc not found"**
— The user-data script didn't complete. SSH in (via SSM session) and re-run `pip install -e '.[neuron,dev]'` in `/home/ubuntu/trnfft`.

**Tests compile but numerical output is off**
— Expected for FP32 on first validation. Check tolerances in test assertions; tighten after baseline is established.
