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

variable "github_repo" {
  description = "GitHub repo in owner/name format, for OIDC trust"
  type        = string
  default     = "scttfrdmn/trnfft"
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

data "aws_ami" "neuron" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI Neuron PyTorch*Ubuntu 22.04*"]
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
  associate_public_ip_address = false

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    cd /home/ubuntu
    sudo -u ubuntu git clone https://github.com/${var.github_repo}.git trnfft
    cd trnfft
    sudo -u ubuntu python3 -m venv .venv
    sudo -u ubuntu .venv/bin/pip install -e '.[neuron,dev]'
  EOF

  tags = {
    Name = var.instance_tag
  }
}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC role
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "${var.instance_tag}-gh-actions"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions" {
  name = "ci-runner"
  role = aws_iam_role.github_actions.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:StartInstances",
          "ec2:StopInstances",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/Name" = var.instance_tag
          }
        }
      },
      {
        Effect   = "Allow"
        Action   = "ec2:DescribeInstances"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:SendCommand",
          "ssm:GetCommandInvocation",
          "ssm:DescribeInstanceInformation",
        ]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_id" {
  value = aws_instance.ci.id
}

output "instance_tag" {
  value       = var.instance_tag
  description = "Set as the Name tag for workflow discovery"
}

output "aws_role_arn" {
  value       = aws_iam_role.github_actions.arn
  description = "Add as AWS_ROLE_ARN repo secret in GitHub"
}

output "aws_region" {
  value       = var.aws_region
  description = "Add as AWS_REGION repo variable in GitHub"
}
