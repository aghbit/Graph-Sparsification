for MODEL in "GCN" "SAGE" "GAT"
do
  for SPARSIFICATION in "LDSDirected" "LSSDirected" "SCANDirected"
  do
    for POWER in 1 2 3 4 5
    do
      python train.py --name "$MODEL-$SPARSIFICATION-$POWER" --dataset cornell --model $MODEL --num_layers 1 --device cuda:0 --sparsification_algorithm $SPARSIFICATION --sparsification_power $POWER
    done
  done
done
