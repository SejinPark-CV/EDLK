CUDA_VISIBLE_DEVICES=0 python train_mnist.py \
  --implementation cuda \
  --connections random \
  --tau 20 \
  --tau-router 1.0 \
  --router-bits 15 \
  --grad-factor 2.0 \
  --channels 256 \
  --K 2 \
  --router-hidden 32 \
  --num-iterations 200000 \
  --batch-size 64 \
  --eval-freq 2000 \
  --learning-rate 0.01 \
  --load-balance-weight 0.01 \
  --seed 0 \
  --curve-csv logs/mnist_edlk_20.csv \
  --save-ckpt checkpoints/mnist_edlk_20.pt


# CUDA_VISIBLE_DEVICES=1 python train_mnist.py \
#   --implementation cuda \
#   --connections random \
#   --tau 25 \
#   --tau-router 1.0 \
#   --router-bits 15 \
#   --grad-factor 2.0 \
#   --channels 256 \
#   --K 2 \
#   --router-hidden 32 \
#   --num-iterations 200000 \
#   --batch-size 64 \
#   --eval-freq 2000 \
#   --learning-rate 0.01 \
#   --load-balance-weight 0.01 \
#   --seed 0 \
#   --curve-csv logs/mnist_edlk_25.csv \
#   --save-ckpt checkpoints/mnist_edlk_25.pt