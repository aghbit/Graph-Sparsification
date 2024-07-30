for SPARSIFICATION in "PreferentialAttachment" "AdjustedRand" "LDS" "LSS" "SCAN"
do
  for POWER in 0.25 0.5 0.75
  do
    python train.py --name "SAGE-$SPARSIFICATION-$POWER" --dataset amazon-ratings --model SAGE --num_layers 2 --device cuda:0 --sparsification_algorithm $SPARSIFICATION --sparsification_power $POWER
  done
done
