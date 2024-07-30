python train.py --name GCN_l1 --dataset cornell --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset cornell --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset cornell --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset cornell --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset cornell --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset cornell --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset cornell --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset cornell --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset cornell --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset cornell --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset cornell --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset cornell --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset cornell --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset cornell --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset cornell --model GAT --num_layers 5 --device cuda:0

