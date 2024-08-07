import re

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from run_ranking_E5 import rank_documents

np.random.seed(42)


def save_dataset(df, file_name, is_val=False):
    # Save to file
    if is_val:
        df2 = df[df[1].str[-1] == '2']
        df4 = df[df[1].str[-1] == '4']
        # df2.sort_values(by=df2.columns[-1]).to_csv("/lv_local/home/niv.b/train_fb_ranker/validation_files/val_features_2.dat", header=False, index=False, sep=' ')
        # df4.sort_values(by=df4.columns[-1]).to_csv("/lv_local/home/niv.b/train_fb_ranker/validation_files/val_features_4.dat", header=False, index=False, sep=' ')
        pd.concat([df4, df2]).sort_values(by=df2.columns[-1]).to_csv(file_name, header=False, index=False, sep=' ')
    else:
        df.sort_values(by=df.columns[-1]).to_csv(file_name, header=False, index=False, sep=' ')


def create_val_summary(val_df, file_name):
    relevant_base_round = int(file_name[-1])
    candidates = list(val_df[val_df.columns[-1]])
    orig_candidates = set([cand.split('$')[0] for cand in candidates])
    df = pd.read_csv('tommy_data.csv')
    rel_df = df[(df['docno'].isin(orig_candidates)) & (df['round_no'] == relevant_base_round)]

    summ_rows = list()
    for idx, row in tqdm(rel_df.iterrows(), total=len(rel_df)):
        top_df = df[(df.round_no == row.round_no) & (df.query_id == row.query_id) & (df.position < row.position)]
        d_cur = list(re.findall(r'.+?[.!?](?:\s+|$)', row["current_document"]))

        for idx_t, row_t in top_df.iterrows():
            # Extract sentences
            g_pool = list(set(filter(lambda s: len(s.split()) >= 2,
                                     list(re.findall(r'.+?[.!?](?:\s+|$)', row_t["current_document"])))))
            for i in range(len(d_cur)):
                for j in range(len(g_pool)):
                    if len(d_cur[i].split()) < 2 or re.match(r'^\W*$', d_cur[i]):
                        continue
                    new_docno = row.docno + "$" + row_t.docno + "_" + str(i + 1) + "_" + str(j + 1)
                    if new_docno not in candidates:
                        continue

                    d_next = d_cur.copy()
                    d_next[i] = g_pool[j]
                    new_text = " ".join(d_next)
                    temp_df = df[(df.round_no == row.round_no + 1) & (df.query_id == row.query_id)].copy()
                    new_row = [new_docno, new_text, relevant_base_round, float('nan'), "switched", row["query"],
                               float('nan'), float('nan')]
                    temp_df.loc[int(temp_df[temp_df.username == row.username].index[0])] = new_row
                    temp_df = rank_documents(row["query"], temp_df, "current_document",
                                             return_embedding=False)

                    rank_promotion_true = row["position"] - \
                                          temp_df[temp_df['username'] == 'switched']["rank"].values[0]
                    summ_rows.append(
                        {"docno": new_docno, "rank": temp_df[temp_df['username'] == 'switched']["rank"].values[0],
                         "rank_promotion": rank_promotion_true,
                         "scaled_rank_promotion": rank_promotion_true / (
                                 row["position"] - 1) if rank_promotion_true > 0 else (
                             rank_promotion_true / (
                                     temp_df['rank'].max() - row["position"]) if rank_promotion_true < 0 else 0),
                         "text": new_text})

    pd.DataFrame(summ_rows).to_csv(f"baseline_dataset_validation_r{relevant_base_round}_summary.csv", index=False)
    print(f"Saved {len(summ_rows)} lines to baseline_dataset_validation_r{relevant_base_round}_summary.csv")


# def save_dataset(features, labels, qids, file_name, is_val=False):
#     # Extract numeric part of qid for sorting
#     qid_numeric = qids.str.extract('(\d+)').astype(int)
#
#     # Create a DataFrame with all components
#     df = pd.concat([labels, qids, features], axis=1)
#
#     # Add the numeric qid for sorting, with a unique column name
#     df['_sort_key'] = qid_numeric
#
#     # Sort by the numeric qid, then drop the sort key column
#     df = df.sort_values(by='_sort_key').drop(columns='_sort_key')
#
#     if is_val:
#         df2 = df[df[1].str[-1] == '2']
#         df5 = df[df[1].str[-1] == '5']
#         df2.to_csv("/lv_local/home/niv.b/train_fb_ranker/validation_files/features_2.dat", header=False, index=False,
#                    sep=' ')
#         df5.to_csv("/lv_local/home/niv.b/train_fb_ranker/validation_files/features_5.dat", header=False, index=False,
#                    sep=' ')
#     else:
#         df.to_csv(file_name, header=False, index=False, sep=' ')


# data = pd.read_csv('/lv_local/home/niv.b/train_fb_ranker/harmonic_features_1', header=None, sep=" ")
file_name = 'baseline_dataset_train_r3'
data = pd.read_csv(f'/lv_local/home/niv.b/train_fb_ranker/{file_name}.txt', header=None, sep=" ")  # TOMMY
labels = data.iloc[:, 0]
qids = data.iloc[:, 1]
features = data.iloc[:, 2:]
features.columns = [f'feature_{i}' for i in range(1, features.shape[1] + 1)]

# X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, stratify=qids, random_state=42)

# train_query_counts = X_train.join(qids).groupby(qids.name).size()
# val_query_counts = X_val.join(qids).groupby(qids.name).size()

df = pd.concat([labels, qids, features], axis=1)

unique_qids = df[1].unique()

unique_qids_2 = df[df[1].str[-1] == '2'][1].unique()
unique_qids_4 = df[df[1].str[-1] == '4'][1].unique()  # TOMMY
val_qids = list(np.random.choice(unique_qids_2, size=int(len(unique_qids_2) * 0.5), replace=False)) + \
           list(np.random.choice(unique_qids_4, size=int(len(unique_qids_4) * 0.5), replace=False))  # TOMMY
train_qids = list(np.setdiff1d(unique_qids, val_qids))

df['sort_key'] = df[1].str.extract('(\d+)').astype(int)

train_df = df[df[1].isin(train_qids)].sort_values('sort_key').drop(columns='sort_key')
val_df = df[df[1].isin(val_qids)].sort_values('sort_key').drop(columns='sort_key')

create_val_summary(val_df, file_name)

save_dataset(train_df, f'/lv_local/home/niv.b/train_fb_ranker/{file_name}_train_set')
save_dataset(val_df, f'/lv_local/home/niv.b/train_fb_ranker/{file_name}_validation_set', is_val=True)

# save_dataset(X_train, y_train, qids.loc[X_train.index], '/lv_local/home/niv.b/train_fb_ranker/harmonic_train_set')
# save_dataset(X_val, y_val, qids.loc[X_val.index], '/lv_local/home/niv.b/train_fb_ranker/harmonic_validation_set',
#              is_val=True)
