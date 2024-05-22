python exp-toy.py --reg_name SIM2 --trainer ERM_SIM
python exp-toy.py --reg_name MIP --trainer ERM
python exp-toy.py  --trainer ERM
python exp-toy.py  --reg_name REX --trainer ERM
python exp-toy.py  --reg_name MRI --trainer ERM
python exp-toy.py  --reg_name IRM --trainer ERM
python exp-toy.py --reg_name MIP --trainer ERM --tune_head #Fishr
python exp-toy.py  --reg_name IBIRM --trainer ERM
python exp-toy.py  --trainer groupDRO
python exp-toy.py --reg_name IDGM --trainer ERM --tune_head --reg_lambda 0.001

python exp-toy-covariate.py --reg_name SIM2 --trainer ERM_SIM
python exp-toy-covariate.py --reg_name MIP --trainer ERM
python exp-toy-covariate.py  --trainer ERM
python exp-toy-covariate.py  --reg_name REX --trainer ERM
python exp-toy-covariate.py  --reg_name MRI --trainer ERM
python exp-toy-covariate.py  --reg_name IRM --trainer ERM
python exp-toy-covariate.py --reg_name MIP --trainer ERM --tune_head #Fishr
python exp-toy-covariate.py  --reg_name IBIRM --trainer ERM
python exp-toy-covariate.py  --trainer groupDRO
python exp-toy-covariate.py --reg_name IDGM --trainer ERM --tune_head --reg_lambda 0.0001

python exp-toy-labelshift.py --reg_name SIM2 --trainer ERM_SIM
python exp-toy-labelshift.py --reg_name MIP --trainer ERM
python exp-toy-labelshift.py  --trainer ERM
python exp-toy-labelshift.py  --reg_name REX --trainer ERM
python exp-toy-labelshift.py  --reg_name MRI --trainer ERM
python exp-toy-labelshift.py  --reg_name IRM --trainer ERM
python exp-toy-labelshift.py --reg_name MIP --trainer ERM --tune_head #Fishr
python exp-toy-labelshift.py  --reg_name IBIRM --trainer ERM
python exp-toy-labelshift.py  --trainer groupDRO
python exp-toy-labelshift.py --reg_name IDGM --trainer ERM --tune_head --reg_lambda 0.0001

python exp-mnist.py --reg_name SIM2 --trainer ERM_SIM
python exp-mnist.py --reg_name MIP --trainer ERM
python exp-mnist.py  --trainer ERM
python exp-mnist.py  --reg_name REX --trainer ERM
python exp-mnist.py  --reg_name MRI --trainer ERM
python exp-mnist.py  --reg_name IRM --trainer ERM
python exp-mnist.py --reg_name MIP --trainer ERM --tune_head #Fishr
python exp-mnist.py  --reg_name IBIRM --trainer ERM
python exp-mnist.py  --trainer groupDRO
python exp-mnist.py --reg_name IDGM --trainer ERM --tune_head --reg_lambda 0.1


