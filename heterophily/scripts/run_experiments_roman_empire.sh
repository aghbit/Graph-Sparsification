for SPARSIFICATION in "PreferentialAttachment" "AdjustedRand" "LDS" "LSS" "SCAN"
do
  for POWER in 1 2 3 4 5
  do
    python train.py --name "GCN-$SPARSIFICATION-$POWER" --dataset roman-empire --model GCN --num_layers 1 --device cuda:0 --sparsification_algorithm $SPARSIFICATION --sparsification_power $POWER
  done
done
