# for seed in 1314 1315 1316
# do
#     python exp_toy_slope_10D.py  --seed=$seed --reg_lambda=0
# done

# for seed in 1314 1315 1316
# do
#     for i in 0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 # 20 50 100
#     do
#         python exp_toy_slope_10D.py --method_name IRM --seed=$seed --trainer ERM --reg_name IRM --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  20 50 100 # 0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 #
#     do
#         python exp_toy_slope_10D.py --method_name MIP --seed=$seed --trainer ERM --reg_name MIP --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 
#     do
#         python exp_toy_slope_10D.py --method_name IB-IRM --seed=$seed --trainer ERM --reg_name IBIRM --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 
#     do
#         python exp_toy_slope_10D.py --method_name REX --seed=$seed --trainer ERM --reg_name REX --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 
#     do
#         python exp_toy_slope_10D.py --method_name MRI --seed=$seed --trainer ERM --reg_name MRI --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100 
#     do
#         python exp_toy_slope_10D.py --method_name IDGM --seed=$seed --trainer ERM --reg_name IDGM --reg_lambda=$i --iteration=100
#     done
# done

# for seed in 1314 1315 1316
# do
#     python exp_toy_slope_10D.py --method_name ERM --seed=$seed --trainer ERM --reg_lambda=0 --iteration=200
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100
#     do
#         python exp_toy_slope_10D.py --method_name Fishr --seed=$seed --trainer ERM --reg_name MIP --reg_lambda=$i --iteration=100 --tune_head
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100
#     do
#         python exp_toy_slope_10D.py --method_name GroupDRO --seed=$seed --trainer GroupDRO  --reg_lambda=$i --iteration=200
#     done
# done

# for seed in 1314 1315 1316
# do
#     for i in  0.01 0.02 0.05 0.1 0.2 0.5 1 2 5 10 20 50 100
#     do
#         python exp_toy_slope_10D.py --method_name CLOvE --seed=$seed --trainer ERM --reg_name CLOvE --reg_lambda=$i --iteration=350
#     done
# done

