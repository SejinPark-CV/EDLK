#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python infer_mnist.py \
  --checkpoint checkpoints/mnist_edlk_25.pt \
  --batch-size 64 \
  --num-workers 4
