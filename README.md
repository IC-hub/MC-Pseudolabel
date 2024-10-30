# Bridging Multicalibration and OOD Generalization Beyond Covariate Shift

Official implemention for the **MC-Pseudolabel** algorithm in the paper ["Bridging Multicalibration and OOD Generalization Beyond Covariate Shift"](https://arxiv.org/abs/2406.00661) by Jiayun Wu, Jiashuo Liu, Peng Cui and Zhiwei Steven Wu.

## Requirements

To install requirements:

```cmd
pip install -r requirements.txt
```

## Single Run

To run MC-Pseudolabel in the paper on PovertyMap, run this command:

```cmd
python exp_poverty_multienv.py --method_name MCPseudolabel --trainer MCPseudolabel
```

The results will be saved in the `results.csv` file, including the following fields:
- timestamp.
- method: MCPseudolabel.
- Pearson_test_average: average Pearsonr between Urban/Rural environments on the test set.
- Pearson_test_worst: worst Pearsonr between Urban/Rural environments on the test set. *The primary metric.*
- Pearson_val_average: average Pearsonr between Urban/Rural environments on the validation set.
- Pearson_val_worst: worst Pearsonr between Urban/Rural environments on the validation set.
- fold: the spit of the dataset (A, B, C, D, E).


## Multiple Runs
We have a script to run the algorithm multiple times across different splits of the PovertyMap dataset and with different seeds, and calculate the average performance and standard deviation based on 3 model selection criteria: in-distribution selection, worst-environment selection, and oracle selection. Please refer to the paper for further details. To reproduce the results for MC-Pseudolabel on PovertyMap in Table 1 of the paper, run this command:

```cmd
python script/script_poverty_multienv.py
``` 

The script support multiple processes and multiple GPUs. Specify the fields `n_jobs` and `cuda_id_list` in the script as needed.

Running this script will overwrite any existing `results.csv` file and generate a new `results_summary.csv` file that reports the final statistics. The file will contain the following fields:
- Method: MCPseudolabel.
- Eval_Pearson_worst: the average of the field `Pearson_test_worst` in `results.csv` across all splits. Within each split, we conduct in-distribution selection. We take into account the best model in terms of the field `Pearson_val_average` in `results.csv`.
- Eval_Pearson_worst_std: the standard deviation of `Eval_Pearson_worst` across all splits.
- Eval_Pearson_average: the average of the field `Pearson_test_average` in `results.csv` across all splits. Within each split, we conduct in-distribution selection.
- Eval_Pearson_average_std: the standard deviation of `Eval_Pearson_average` across all splits.
- *_WorstSelection: statistics under worst-environment selection. We take into account the best model in terms of the field `Pearson_val_worst` in `results.csv`.
- *_OracleSelection: statistics under oracle selection. We take into account the best model in terms of the field `Pearson_test_worst` in `results.csv`.


