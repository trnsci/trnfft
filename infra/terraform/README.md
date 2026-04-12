# Terraform — trnfft CI Instance

Provisions an EC2 Trainium/Inferentia instance for running `pytest -m neuron`
from GitHub Actions via SSM. The instance stays stopped when not in use; the
workflow starts it, runs tests, and stops it again.

## What it creates

- **EC2 instance** — `trn1.2xlarge` by default (Deep Learning AMI Neuron, Ubuntu 22.04)
- **IAM instance profile** — SSM managed instance core
- **Security group** — no inbound, outbound only (SSM uses VPC endpoints or NAT)
- **GitHub Actions OIDC provider + role** — lets the workflow assume a role with
  permission to start/stop and send SSM commands to *this specific instance*

## Apply

```bash
cd infra/terraform

terraform init

terraform plan \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"

terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"
```

Outputs will include `aws_role_arn` and `aws_region`. Add these to your repo:

- **Secret** (`Settings → Secrets and variables → Actions → Secrets`): `AWS_ROLE_ARN` = `<output>`
- **Variable** (`Settings → Secrets and variables → Actions → Variables`): `AWS_REGION` = `<output>`

## Customization

| Variable | Default | Notes |
|----------|---------|-------|
| `aws_region` | `us-east-1` | Trainium available in us-east-1, us-west-2, eu-west-1 |
| `instance_type` | `trn1.2xlarge` | Alternatives: `trn2.8xlarge`, `inf2.xlarge` |
| `instance_tag` | `trnfft-ci-trn1` | Must match the `inputs.instance_type` in `neuron.yml` |
| `github_repo` | `scttfrdmn/trnfft` | Used in OIDC trust policy |

For multiple instance types, apply the module multiple times with different
`instance_type` and `instance_tag`.

## Cost estimate

On-demand pricing (us-east-1, as of 2026):

| Type | Hourly | Typical CI run (10 min) |
|------|-------:|-------------------------:|
| trn1.2xlarge | ~$1.34 | ~$0.22 |
| trn2.8xlarge | ~$10.00 | ~$1.67 |
| inf2.xlarge | ~$0.76 | ~$0.13 |

Idle cost when stopped: **$0** for compute (only EBS storage ~$10/mo for 100 GB gp3).

## Trigger the workflow

```bash
gh workflow run neuron.yml -R scttfrdmn/trnfft -f instance_type=trn1
```

Or via the Actions tab → "Neuron Hardware Tests" → "Run workflow".
