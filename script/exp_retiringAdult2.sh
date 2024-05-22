for seed in 1316 # 1315 1316
do
    for i in  0.017  0.106  0.287 0.392 0.41  0.647 0.83 3.161 3.435 3.851  4.07  7.535  8.371  9.877  20.981  23.409  25.407  40.438  58.623  73.42
    do
        python exp_retiringAdult2.py --method_name CLOvE --seed=$seed --trainer ERM --reg_name CLOvE --reg_lambda=$i 
    done
done