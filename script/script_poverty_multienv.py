import os
import multiprocessing
import random
import subprocess
import pandas as pd

random.seed(1314)

n_jobs = 4

seed_list = [1314, 1316, 1317]
fold_list = ['A', 'B', 'C', 'D', 'E']
learning_rate_list = [0.001]
batch_size_list = [64]
cuda_id_list = [1, 2, 3, 4]


os.system("rm results.csv")


def run_script(params):
    cmd = ['python', 'exp_poverty_multienv.py', '--method_name', 'MCPseudolabel', '--trainer', 'MCPseudolabel']
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
            'cuda':f'cuda:{cuda_id_list[(i*3+j) % len(cuda_id_list)]}'
        }
        # run_script(params)
        pool.apply_async(run_script, args=(params,))
pool.close()
pool.join()


df = pd.read_csv("./results.csv", header=0)
result_df = pd.DataFrame(columns=[
    'Method', 
    'Eval_Pearson_worst', 'Eval_Pearson_worst_std',
    'Eval_Pearson_average', 'Eval_Pearson_average_std',
    'Eval_Pearson_worst_WorstSelection', 'Eval_Pearson_worst_std_WorstSelection',
    'Eval_Pearson_average_WorstSelection', 'Eval_Pearson_average_std_WorstSelection',
    'Eval_Pearson_worst_OracleSelection', 'Eval_Pearson_worst_std_OracleSelection',
    'Eval_Pearson_average_OracleSelection', 'Eval_Pearson_average_std_OracleSelection',
    'Val_Pearson','Val_Pearson_std'
])
record_df = pd.DataFrame(columns=df.columns)
record_df_worst = pd.DataFrame(columns=df.columns)
record_df_oracle = pd.DataFrame(columns=df.columns)
for method in df['method'].unique():
    df_method = df[df.method==method].reset_index(drop=True)
    for fold in df_method['fold'].unique():
        df_select = df_method[df_method['fold']==fold].reset_index(drop=True)
        df_select_val = df_select.iloc[df_select['Pearson_val_average'].argmax()] # val_metric
        df_select_worst = df_select.iloc[df_select['Pearson_val_worst'].argmax()]
        df_select_oracle = df_select.iloc[df_select['Pearson_test_worst'].argmax()]
        record_df = record_df.append(df_select_val)
        record_df_worst = record_df_worst.append(df_select_worst)
        record_df_oracle = record_df_oracle.append(df_select_oracle)
    result_df = result_df.append({
        'Method':method,
        
        'Eval_Pearson_worst':record_df[record_df['method']==method]['Pearson_test_worst'].mean(),
        'Eval_Pearson_worst_std':record_df[record_df['method']==method]['Pearson_test_worst'].std(),
        'Eval_Pearson_average':record_df[record_df['method']==method]['Pearson_test_average'].mean(),
        'Eval_Pearson_average_std':record_df[record_df['method']==method]['Pearson_test_average'].std(),
        
        'Eval_Pearson_worst_WorstSelection':record_df_worst[record_df_worst['method']==method]['Pearson_test_worst'].mean(),
        'Eval_Pearson_worst_std_WorstSelection':record_df_worst[record_df_worst['method']==method]['Pearson_test_worst'].std(),
        'Eval_Pearson_average_WorstSelection':record_df_worst[record_df_worst['method']==method]['Pearson_test_average'].mean(),
        'Eval_Pearson_average_std_WorstSelection':record_df_worst[record_df_worst['method']==method]['Pearson_test_average'].std(),
        
        'Eval_Pearson_worst_OracleSelection':record_df_oracle[record_df_oracle['method']==method]['Pearson_test_worst'].mean(),
        'Eval_Pearson_worst_std_OracleSelection':record_df_oracle[record_df_oracle['method']==method]['Pearson_test_worst'].std(),
        'Eval_Pearson_average_OracleSelection':record_df_oracle[record_df_oracle['method']==method]['Pearson_test_average'].mean(),
        'Eval_Pearson_average_std_OracleSelection':record_df_oracle[record_df_oracle['method']==method]['Pearson_test_average'].std(),
        
        'Val_Pearson':record_df[record_df['method']==method]['Pearson_val_average'].mean(),
        'Val_Pearson_std':record_df[record_df['method']==method]['Pearson_val_average'].std(),
    }, ignore_index=True)
result_df.sort_values(['Eval_Pearson_worst'], ascending=False)
result_df.to_csv("result_summary.csv", index=False)
