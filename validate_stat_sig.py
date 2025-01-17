import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import permutation_test
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")


def create_initial():
    filepath = "/lv_local/home/niv.b/train_fb_ranker/output_results/baseline_model_choice_results_full_sig.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    trait_df = pd.read_csv("./output_results/trait_summary.csv")
    res_df = pd.read_csv("./output_results/baseline_model_choice_results_full.csv")
    res_df[['name', 'metric']] = res_df['model'].str.split("#", expand=True)
    res_df = res_df.merge(trait_df, on='name', how='left')[[
        'model', 'name', 'metric', 'tree', 'leaf', 'shrinkage', 'rank_test',
        'rank_promotion_test', 'scaled_rank_promotion_test', 'rank_val',
        'rank_promotion_val', 'scaled_rank_promotion_val']]
    res_df["stat_sig"] = np.nan
    res_df["stat_sig_seed"] = np.nan
    return res_df


def get_bot_vals(bot_file, bot_name):
    return pd.read_csv(f"./output_results/{bot_file}").query("username == @bot_name")['scaled_rank_promotion'].values


def mean_statistic(x, y):
    return np.mean(x) - np.mean(y)


def process_model(model_name, path_dict):
    df = pd.read_csv(path_dict["summary"])
    df["orig_docno"] = df['docno'].str.split("$").str[0]
    score_path = f'/lv_local/home/niv.b/train_fb_ranker/output_results/ranker_test_results/predictions_baseline_model_{model_name}_test.txt'
    df["score"] = pd.read_csv(score_path, header=None, sep='\t', usecols=[2])
    if df["score"].isna().any():
        print(f"ERROR in {model_name}: score column contains NaN values")
    df['position'] = df.groupby("orig_docno")['score'].rank(method='first', ascending=False).astype(int)
    return df[df.position == 1]['scaled_rank_promotion'].values


def worker(args):
    seed, idx, row, bot_values, test_paths, student_path = args
    if row.scaled_rank_promotion_test > 0:
        return {'idx': idx, 'stat_sig': 0, 'stat_sig_seed': None, 'pvalue': None}
    if row.stat_sig == 1:
        return {'idx': idx, 'stat_sig': 1, 'stat_sig_seed': row.stat_sig_seed, 'pvalue': row.pvalue}
    baseline_values = process_model(row.model, test_paths)
    res = permutation_test(
        (bot_values, baseline_values),
        mean_statistic,
        n_resamples=100000,
        alternative='two-sided',
        random_state=seed
    )
    is_sig = res.pvalue <= alpha
    if is_sig:
        
        student_values = pd.read_csv(f"./output_results/{student_path}")['scaled_pos_diff'].values
        res_st = permutation_test(
            (bot_values, student_values),
            mean_statistic,
            n_resamples=100000,
            alternative='two-sided',
            random_state=seed
        )
        is_sig_st = res_st.pvalue <= alpha
        print(f"BASELINE: Model {row.model} - Rejected Null Hypothesis with PVAL {res.pvalue:.3f}! and {res_st.pvalue:.3f} student pval; Seed: {seed}")
        is_sig = is_sig and is_sig_st
        if is_sig:
            print(f"STUDENT: Model {row.model} - Rejected Null Hypothesis with PVAL {res.pvalue:.3f}! Seed: {seed}")

    return {'idx': idx, 'stat_sig': int(is_sig), 'stat_sig_seed': seed if is_sig else None,
            'pvalue': res.pvalue if is_sig else None}


alpha = 0.05
bot_name = "LIW_1201"
temp = 100
round_ = 56

test_paths = {"embeddings": f"baseline_dataset_test_r{round_}.txt",
              "summary": f"baseline_dataset_test_r{round_}_summary.csv"}
student_path = f"{bot_name}.csv"

bot_values = get_bot_vals(bot_file=f"test_tommy{round_}@F{temp}.csv", bot_name=f"{bot_name}@{temp}")  # for 56!
res_df_full = create_initial()
NUMBER_OF_WORKERS = max(1, os.cpu_count())

for seed in range(24, 10000):
    print(f"Processing seed {seed}")
    tasks = [(seed, idx, row, bot_values, test_paths, student_path) for idx, row in res_df_full.iterrows()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        results = []
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(f.result())

    for res in results:
        idx = res['idx']
        res_df_full.at[idx, 'stat_sig'] = res['stat_sig']
        if res['stat_sig_seed']:
            res_df_full.at[idx, 'stat_sig_seed'] = res['stat_sig_seed']
            res_df_full.at[idx, 'pvalue'] = res['pvalue']
    res_df_full.to_csv(
        "/lv_local/home/niv.b/train_fb_ranker/output_results/baseline_model_choice_results_full_sig.csv",
        index=False
    )
    with open("last_seed_finished.txt", 'w') as f:
        f.write(str(seed))
        f.close()
