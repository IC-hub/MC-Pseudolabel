# for seed in 1317 1318 1319 1320 1321 1322
# do
#     python exp_retiringAdult.py  --seed=$seed --reg_lambda=0 
# done

# for seed in 1315 # 1316
# do
#     for i in 0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name IRM --seed=$seed --trainer ERM --reg_name IRM --reg_lambda=$i
#     done
# done

# for seed in 1314 
# do
#     for i in 0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name MIP --seed=$seed --trainer ERM --reg_name MIP --reg_lambda=$i 
#     done
# done

# for seed in 1314
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name IB-IRM --seed=$seed --trainer ERM --reg_name IBIRM --reg_lambda=$i 
#     done
# done

# for seed in 1314
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name REX --seed=$seed --trainer ERM --reg_name REX --reg_lambda=$i 
#     done
# done

# for seed in 1314 
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name MRI --seed=$seed --trainer ERM --reg_name MRI --reg_lambda=$i 
#     done
# done

# for seed in 1314 # 1315 1316
# do
#     for i in 0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name IDGM --seed=$seed --trainer ERM --reg_name IDGM --reg_lambda=$i 
#     done
# done

for seed in 1317 1318 1319 1320 1321 1322
do
    python exp_retiringAdult.py --method_name ERM --seed=$seed --trainer ERM --reg_lambda=0 --iteration=200
done

# for seed in 1314 # 1315 1316
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name Fishr --seed=$seed --trainer ERM --reg_name MIP --reg_lambda=$i  --tune_head
#     done
# done

# for seed in 1314 # 1315 1316
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name GroupDRO --seed=$seed --trainer GroupDRO  --reg_lambda=$i 
#     done
# done

# for seed in 1314 # 1315 1316
# do
#     for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
#     do
#         python exp_retiringAdult.py --method_name CLOvE --seed=$seed --trainer ERM --reg_name CLOvE --reg_lambda=$i 
#     done
# done

# for seed in 1315 1316
# do
#     python exp_retiringAdult.py --method_name Oracle --seed=$seed --trainer ERM --reg_lambda=0 --iteration=600 --dataset_path=./data/RetiringAdult/adult/processed_occ_MIL
# done
