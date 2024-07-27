for SPARSIFICATION in "PreferentialAttachment" "AdjustedRand" "LDS" "LSS" "SCAN"
do
  for POWER in 1 2 3 4 5
  do
    python train.py --name "SAGE-$SPARSIFICATION-$POWER" --dataset amazon-ratings --model SAGE --num_layers 5 --device cuda:0 --sparsification_algorithm $SPARSIFICATION --sparsification_power $POWER
  done
done
