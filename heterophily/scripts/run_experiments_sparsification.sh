for SPARSIFICATION in "Jaccard" "PreferentialAttachment" "AdjustedRand" "LDS" "LSS" "SCAN"
do
  for POWER in 2 4 6 8 10
  do
    python train.py --name "SAGE-$SPARSIFICATION-$POWER" --dataset roman-empire --model SAGE --num_layers 1 --device cuda:0 --sparsification_algorithm $SPARSIFICATION --sparsification_power $POWER
  done
done
