import multiprocessing
import random
import subprocess

random.seed(1314)

n_jobs = 4


seed_list = [1314, 1316, 1317]
fold_list = ['A', 'B', 'C', 'D', 'E']
learning_rate_list = [0.001]
batch_size_list = [64]
reg_lambda_list = [0.017, 0.106, 0.287, 0.392, 0.41, 0.647, 0.83, 3.161, 3.435, 3.851, 4.07, 7.535, 8.371, 9.877, 20.981, 23.409, 25.407, 40.438, 58.623, 73.42]


# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'MCDeboost', '--trainer', 'MCDeboost']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=15)
# for i in range(3):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'ERM', '--trainer', 'ERM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(3):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()


# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'IRM', '--trainer', 'ERM', '--reg_name', 'IRM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'GroupDRO', '--trainer', 'GroupDRO']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()


# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'IB-IRM', '--trainer', 'ERM', '--reg_name', 'IBIRM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'REX', '--trainer', 'ERM', '--reg_name', 'REX']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold,
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'MIP', '--trainer', 'ERM', '--reg_name', 'MIP']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold, 
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()


# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'MRI', '--trainer', 'ERM', '--reg_name', 'MRI']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=5)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold, 
#             'cuda':f'cuda:{(2+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()


# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'CLOvE', '--trainer', 'ERM', '--reg_name', 'CLOvE']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=20)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold, 
#             'cuda':f'cuda:{(3+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'Fishr', '--trainer', 'ERM', '--reg_name', 'MIP', '--tune_head']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=20)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold, 
#             'cuda':f'cuda:{(3+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'IDGM', '--trainer', 'ERM', '--reg_name', 'IDGM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=4)
# for i in range(12):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     seed = seed_list[i % len(seed_list)]
#     reg_lambda = random.choice(reg_lambda_list)
#     for j, fold in enumerate(fold_list[:1]):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': reg_lambda,
#             'dataset_fold': fold, 
#             'cuda':f'cuda:{(3+j)%8}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

def run_script(params):
    cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'Oracle', '--trainer', 'ERM']
    for key, val in params.items():
        cmd += [f'--{key}', f'{val}']
    subprocess.run(cmd)
pool = multiprocessing.Pool(processes=15)
for i in range(3):
    learning_rate = random.choice(learning_rate_list)
    batch_size = random.choice(batch_size_list)
    seed = seed_list[i % len(seed_list)]
    for j, fold in enumerate(fold_list):
        params = {
            'seed': seed,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'dataset_fold': fold,
            'cuda':f'cuda:{(2+j)%8}'
        }
        # run_script(params)
        pool.apply_async(run_script, args=(params,))
pool.close()
pool.join()