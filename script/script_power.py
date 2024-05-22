import multiprocessing
import random
import subprocess

random.seed(1314)

n_jobs = 4


# seed_list = [1314, 1315, 1316]
seed_list = [1317, 1318, 1319, 1320, 1321, 1322]
learning_rate_list = [0.001, 0.005, 0.01, 0.05]
# learning_rate_list = [0.05]
batch_size_list = [256, 512, 1024, 2048]
alpha_threshold_list=[0.1, 0.2, 0.5, 0.7]
lambda_up_list=[5, 10, 20, 50]
eta_list=[0.2, 0.5, 1, 1.5]
t_list = [0.1, 0.5, 1, 5, 10, 50, 100, 200]
# t_list = [0.5]

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'MCDeboost', '--trainer', 'MCDeboost']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=4)
# for _ in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'cuda':f'cuda:{1+i}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'ERM', '--trainer', 'ERM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=4)
# for _ in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'cuda':f'cuda:{1+i}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

def run_script(params):
    cmd = ['python', 'exp_power.py', '--method_name', 'JTT', '--trainer', 'JTT']
    for key, val in params.items():
        cmd += [f'--{key}', f'{val}']
    subprocess.run(cmd)
pool = multiprocessing.Pool(processes=n_jobs)
for j in range(20):
    learning_rate = random.choice(learning_rate_list)
    batch_size = random.choice(batch_size_list)
    alpha_threshold = random.choice(alpha_threshold_list)
    lambda_up = random.choice(lambda_up_list)
    for i, seed in enumerate(seed_list):
        params = {
            'seed': seed,
            'iteration': 225,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'alpha_threshold': alpha_threshold,
            'jtt_lambda_up': lambda_up,
            'cuda':f'cuda:{(j*len(seed_list)+i)%n_jobs}'
        }
        # run_script(params)
        pool.apply_async(run_script, args=(params,))
pool.close()
pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'CVaR', '--trainer', 'CVaR']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=n_jobs)
# for j in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     alpha_threshold = random.choice(alpha_threshold_list)
#     lambda_up = random.choice(lambda_up_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'iteration': 225,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'alpha_threshold': alpha_threshold,
#             'cuda':f'cuda:{(j*len(seed_list)+i)%n_jobs}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'X2DRO', '--trainer', 'X2DRO']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=n_jobs)
# for j in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     eta = random.choice(eta_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'iteration': 225,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': eta,
#             'cuda':f'cuda:{(j*len(seed_list)+i)%n_jobs}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'TERM', '--trainer', 'TERM']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=n_jobs)
# for j in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     t = random.choice(t_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'iteration': 225,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'reg_lambda': t,
#             'cuda':f'cuda:{(j*len(seed_list)+i)%n_jobs}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'Oracle', '--trainer', 'ERM', '--dataset_path', './data/power/processed_test']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=4)
# for _ in range(8):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'cuda':f'cuda:{1+i}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()

# mixup_alpha_list = [0.5, 1, 1.5, 2]
# mixup_sigma_list = [0.01, 0.1, 1, 10, 100]

# def run_script(params):
#     cmd = ['python', 'exp_power.py', '--method_name', 'CMixup', '--trainer', 'CMixup']
#     for key, val in params.items():
#         cmd += [f'--{key}', f'{val}']
#     subprocess.run(cmd)
# pool = multiprocessing.Pool(processes=n_jobs)
# for j in range(20):
#     learning_rate = random.choice(learning_rate_list)
#     batch_size = random.choice(batch_size_list)
#     mixup_alpha = random.choice(mixup_alpha_list)
#     mixup_sigma = random.choice(mixup_sigma_list)
#     for i, seed in enumerate(seed_list):
#         params = {
#             'seed': seed,
#             'iteration': 225,
#             'learning_rate': learning_rate,
#             'batch_size': batch_size,
#             'mixup_alpha': mixup_alpha,
#             'mixup_sigma': mixup_sigma,
#             'cuda':f'cuda:{(j*len(seed_list)+i)%n_jobs}'
#         }
#         # run_script(params)
#         pool.apply_async(run_script, args=(params,))
# pool.close()
# pool.join()