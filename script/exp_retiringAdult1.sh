for seed in 1314 1315 1316
do
    python exp_retiringAdult.py --method_name Oracle --seed=$seed --trainer ERM --reg_lambda=0 --iteration=200 --dataset_path=./data/RetiringAdult/adult/processed_occ_MIL
done