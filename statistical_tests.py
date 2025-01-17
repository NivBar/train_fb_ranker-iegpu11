import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import permutation_test
from itertools import combinations
import pickle
from tqdm import tqdm
import os
from validate_stat_sig import process_model




def mean_statistic(x, y):
    return np.mean(x) - np.mean(y)


round_no = 56

rel_cols = ['rank', 'rank_promotion', 'scaled_rank_promotion', 'CF@1', 'NCF@1',
            'EF@10_dense', 'EF@10_sparse', 'EF@10_max_dense', 'EF@10_max_sparse',
            'NEF@10_dense', 'NEF@10_sparse', 'NEF@10_max_dense', 'NEF@10_max_sparse',
            'EF@5_dense', 'EF@5_sparse', 'EF@5_max_dense', 'EF@5_max_sparse',
            'NEF@5_dense', 'NEF@5_sparse', 'NEF@5_max_dense', 'NEF@5_max_sparse']

files_dir = "./output_results/stat_sig_files/"
test_paths = {"embeddings": "baseline_dataset_test_r56.txt", "summary": "baseline_dataset_test_r56_summary.csv"}
###########################


df = pd.read_csv(files_dir + f"test_tommy56@F100.csv").dropna(how='any', axis=1)[['tag'] + rel_cols]
x = np.array(list(df[df.tag == 'listwise']['scaled_rank_promotion'].values))
y_model = process_model('LM590#RR@1', test_paths)
y = np.array(list(df[df.tag == 'LambdaMART_baseline']['scaled_rank_promotion'].values))

for seed in tqdm(range(1,10000)):
    result = permutation_test((x, y), mean_statistic, n_resamples=100000, alternative='two-sided', random_state=seed)
    if result.pvalue <= 0.05:
        print(f"DONE! pvalue {result.pvalue}, seed {seed}")


###########################

# dfs = []
#
# for file in [f for f in os.listdir(files_dir) if f.startswith("test_")]:
#     temp = file.split("@F")[1].split(".")[0]
#     df = pd.read_csv(files_dir + file)[rel_cols + ['tag']].rename(
#         {'rank': 'average rank', 'rank_promotion': 'raw promotion', 'scaled_rank_promotion': 'scaled promotion'},
#         axis=1)
#     df = df.groupby('tag').mean().sort_values('scaled promotion', ascending=False).reset_index()
#     df['temperature'] = np.where(df['tag'].isin(['LM_baseline']), np.nan, int(temp) / 100)
#     dfs.append(df)
#
# res = pd.concat(dfs).sort_values(['tag', 'temperature'], ascending=[True, False]).drop_duplicates().reset_index(
#     drop=True)
# res = res.loc[res.drop(['temperature'], axis=1).drop_duplicates().index]
#
# temperatures = np.sort(res.temperature.unique())[::-1]
# temps = [0.0, 0.5, 1.0, 1.5, 2.0]
# combs = list(combinations(res.tag.unique(), 2))
# combs = [c for c in combs if 'LambdaMART_baseline' in c]
# alpha = 0.05
#
# if os.path.exists(files_dir + 'p_values.pkl'):
#     with open(files_dir + 'p_values.pkl', 'rb') as f:
#         p_vals_dict = pickle.load(f)
# else:
#     p_vals_dict = dict()
#
# for temp in temperatures:
#     cp = f"tommy{round_no}@F{str(int(temp * 100))}"
#     df = pd.read_csv(files_dir + f"test_{cp}.csv").dropna(how='any', axis=1)[['tag'] + rel_cols]
#     # df = pd.concat([df, create_student_df(cp)])
#     # print(f"\n\n### Temperature = {temp}")
#     for col in rel_cols:
#         # print(f"\n##### Column = {col}\n")
#         for comb in tqdm(combs, total=len(combs), desc=f'Temp: {temp}, Col: {col}'):
#             if (temp, col, comb[0], comb[1]) in p_vals_dict:
#                 val = p_vals_dict[(temp, col, comb[0], comb[1])]
#                 # print(f"####### Combination: {comb}, P-value: {val}, Reject (Bonferroni): {val <= bon_alpha}")
#                 continue
#             elif 'student' in comb:
#                 continue
#             elif comb[0] != 'LambdaMART_baseline':
#                 continue
#
#             # TODO: remove!!!
#             col = 'scaled_rank_promotion'
#
#             y = np.array(list(df[df.tag == comb[0]][col].values))
#             x = np.array(list(df[df.tag == comb[1]][col].values))
#             try:
#                 result = permutation_test((x, y), mean_statistic, n_resamples=100000, alternative='two-sided',
#                                           random_state=1)
#             except Exception as e:
#                 print(e)
#                 print(f"ERROR!! Combination: {comb}")
#
#             p_vals_dict[(temp, col, comb[0], comb[1])] = result.pvalue
#             p_vals_dict[(temp, col, comb[1], comb[0])] = result.pvalue
#
#         with open('p_values.pkl', 'wb') as f:
#             pickle.dump(p_vals_dict, f)