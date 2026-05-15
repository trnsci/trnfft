terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region for the CI instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "trn1.2xlarge"
  # Other options: trn2.8xlarge (~$10/hr), inf2.xlarge (~$0.76/hr)
}

variable "instance_tag" {
  description = "Tag used by neuron.yml workflow to find the instance"
  type        = string
  default     = "trnfft-ci-trn1"
}

variable "use_spot" {
  description = "Use spot instances (cheaper, may be interrupted)"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum spot price (on-demand price used if empty)"
  type        = string
  default     = ""
}

variable "vpc_id" {
  description = "VPC to place the instance in"
  type        = string
}

variable "subnet_id" {
  description = "Subnet for the instance (public or private with NAT)"
  type        = string
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Deep Learning AMI with Neuron SDK pre-installed
# ---------------------------------------------------------------------------

# NKI kernels are validated against neuronxcc 2.24.x shipped on
# Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) — version 20260410
# or later. See docs/installation.md for the compatibility matrix.
data "aws_ami" "neuron" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI Neuron PyTorch 2.9*Ubuntu 24.04*"]
  }
}

# ---------------------------------------------------------------------------
# IAM role for the EC2 instance (SSM access)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "instance" {
  name = "${var.instance_tag}-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.instance_tag}-profile"
  role = aws_iam_role.instance.name
}

# ---------------------------------------------------------------------------
# Security group (SSM only, no inbound)
# ---------------------------------------------------------------------------

resource "aws_security_group" "instance" {
  name        = "${var.instance_tag}-sg"
  description = "SSM-only access for trnfft CI"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance
# ---------------------------------------------------------------------------

resource "aws_instance" "ci" {
  ami                         = data.aws_ami.neuron.id
  instance_type               = var.instance_type
  subnet_id                   = var.subnet_id
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  vpc_security_group_ids      = [aws_security_group.instance.id]
  associate_public_ip_address = true  # Needed for SSM agent to reach regional endpoint without VPC endpoints

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price != "" ? var.spot_max_price : null
        instance_interruption_behavior = "terminate"
        spot_instance_type             = "one-time"
      }
    }
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    cd /home/ubuntu
    sudo -u ubuntu git clone https://github.com/trnsci/trnfft.git trnfft
    # Install into the AMI's pre-built Neuron venv (has neuronxcc preinstalled).
    # Use [dev] only — [neuron] would try to fetch neuronxcc from PyPI where it doesn't exist.
    NEURON_VENV=$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1)
    sudo -u ubuntu $NEURON_VENV/bin/pip install -e '/home/ubuntu/trnfft[dev]'
  EOF

  tags = {
    Name = var.instance_tag
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_id" {
  value = aws_instance.ci.id
}

output "instance_tag" {
  value       = var.instance_tag
  description = "Name tag used by scripts/run_neuron_tests.sh"
}

output "aws_region" {
  value       = var.aws_region
  description = "Region to pass to AWS CLI commands"
}
