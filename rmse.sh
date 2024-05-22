for i in 0.1 0.2 0.5 1 2 5 10 20 50 100
do
    python exp_toy_slope.py --reg_lambda=$i --learning_rate=0.5 --iteration=200
done