import shutil
from itertools import combinations, product
import pandas as pd
import os
from tqdm import tqdm
import numpy as np


def lines_to_dict(lines, name):
    metric = name.split('#')[-1]
    res = {"username": name, "metric": metric}
    lines = [line.replace('#', '').strip().split("=") for line in lines]
    for k, v in lines[1:]:
        key, value = k.strip(), float(v.strip())
        res[key] = value
    return res


def create_model_df():
    rows = []
    for file in os.listdir('/lv_local/home/niv.b/train_fb_ranker/trained_models/'):
        if file.startswith('baseline'):
            name = file.split('_')[-1]
            with open(f"/lv_local/home/niv.b/train_fb_ranker/trained_models/{file}") as input_file:
                lines = []
                while True:
                    line = input_file.readline()
                    if line == '\n':
                        break
                    lines.append(line)
                row = lines_to_dict(lines, name)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def process_combination(comb, rel_val, unique_T, unique_L, unique_LR):
    rm_t, rm_l, rm_lr = comb
    min_param_no = 3

    temp_t = [x for x in unique_T if x not in rm_t]
    temp_l = [x for x in unique_L if x not in rm_l]
    temp_lr = [x for x in unique_LR if x not in rm_lr]

    if len(temp_lr) < min_param_no or len(temp_l) < min_param_no or len(temp_t) < min_param_no:
        return None

    temp_df = rel_val[(rel_val['No. of trees'].isin(temp_t)) & (rel_val['No. of leaves'].isin(temp_l)) & (
        rel_val['Learning rate'].isin(temp_lr))]

    if temp_df.shape[0] == 0:
        temp_t = sorted([int(x) for x in temp_t])
        temp_l = sorted([int(x) for x in temp_l])
        temp_lr = sorted([float(x) for x in temp_lr])
        row = {"total_params": len(temp_t) + len(temp_l) + len(temp_lr),
               "min_qua": min([len(temp_t), len(temp_l), len(temp_lr)]),
               "variance": np.round(np.var([len(temp_t), len(temp_l), len(temp_lr)]), 3), "T": temp_t, "L": temp_l,
               "LR": temp_lr}

        # print(row)
        return row

    else:
        return None


def create_val_test_param_dfs():
    param_df = create_model_df()
    param_df.to_csv("/lv_local/home/niv.b/train_fb_ranker/trained_models/model_param_summary.csv", index=False)

    val_df = pd.read_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/best_val_bots.csv")
    val_df["metric"] = val_df.username.apply(lambda x: x.split("#")[1])
    val_df["username"] = val_df.username.apply(lambda x: x.split("_")[1])
    val_df = val_df.merge(param_df, on="username", how="left", suffixes=(None, "_y")).dropna()

    test_df = pd.read_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/best_test_bots.csv")
    test_df["metric"] = test_df.username.apply(lambda x: x.split("#")[-1])
    test_df["username"] = test_df.username.apply(lambda x: x.split("_")[1])
    test_df = test_df.merge(param_df, on="username", how="left", suffixes=(None, "_y")).dropna()
    return val_df, test_df, param_df


def calculate_combinations(val_df, param_df, candidate_name, metric):
    # Retrieve candidate specific values
    candidate_info = param_df[param_df.username == candidate_name]
    candidate_trees = candidate_info['No. of trees'].values[0]
    candidate_leaves = candidate_info['No. of leaves'].values[0]
    candidate_lr = candidate_info['Learning rate'].values[0]

    # Filter and reset index for relevant validation data
    rel_val = val_df[val_df.metric == metric].reset_index(drop=True)
    curr_ind = rel_val[rel_val.username == candidate_name].index[0]
    rel_val = rel_val.head(curr_ind)

    # Function to sort values based on frequency
    def sort_by_frequency(column, exclude_value):
        counts = rel_val[column].value_counts()
        sorted_values = counts.index.tolist()
        if exclude_value in sorted_values:
            sorted_values.remove(exclude_value)
        return sorted_values

    # Sort Trees, Leaves, and LR by frequency
    Trees = sort_by_frequency('No. of trees', candidate_trees)
    Leaves = sort_by_frequency('No. of leaves', candidate_leaves)
    LR = sort_by_frequency('Learning rate', candidate_lr)

    # Generate combinations
    combinations_list1 = [combo for r in range(len(Trees) + 1) for combo in combinations(Trees, r)]
    combinations_list2 = [combo for r in range(len(Leaves) + 1) for combo in combinations(Leaves, r)]
    combinations_list3 = [combo for r in range(len(LR) + 1) for combo in combinations(LR, r)]

    # Combine the combinations from all three lists
    all_combinations = list(product(combinations_list1, combinations_list2, combinations_list3))
    all_combinations = sorted(all_combinations, key=lambda x: len(x[0]) + len(x[1]) + len(x[2]), reverse=True)
    return all_combinations, rel_val


# def calculate_combinations(val_df, param_df):
#     candidate_trees = param_df[param_df.username == candidate_name]['No. of trees'].values[0]
#     candidate_leaves = param_df[param_df.username == candidate_name]['No. of leaves'].values[0]
#     candidate_lr = param_df[param_df.username == candidate_name]['Learning rate'].values[0]
#
#     rel_val = val_df[(val_df.metric == metric)].reset_index(drop=True)
#     curr_ind = rel_val[rel_val.username == candidate_name].index[0]
#     rel_val = rel_val.head(curr_ind)
#
#     Trees = [x for x in rel_val['No. of trees'].unique() if x != candidate_trees]
#     Leaves = [x for x in rel_val['No. of leaves'].unique() if x != candidate_leaves]
#     LR = [x for x in rel_val['Learning rate'].unique() if x != candidate_lr]
#
#     combinations_list1 = [combo for r in range(len(Trees) + 1) for combo in combinations(Trees, r)]
#     combinations_list2 = [combo for r in range(len(Leaves) + 1) for combo in combinations(Leaves, r)]
#     combinations_list3 = [combo for r in range(len(LR) + 1) for combo in combinations(LR, r)]
#
#     # Combine the combinations from all three lists
#     all_combinations = list(product(combinations_list1, combinations_list2, combinations_list3))
#     return all_combinations, rel_val


def add_candidate_value(param_df, lst, col, candidate_name):
    lst = [float(x) for x in lst]
    candidate_value = param_df[param_df.username == candidate_name][col].values[0]
    if candidate_value not in lst:
        lst = np.append(lst, float(candidate_value))
    return sorted(lst)


def save_chosen(param_df, chosen_t, chosen_l, chosen_lr, metric, candidate_name):

    bf_df = pd.read_csv(f"/lv_local/home/niv.b/train_fb_ranker/output_results/archive/bot_followup_FULL.csv")
    bf_df = bf_df[bf_df.username == candidate_name]
    bf_df['username'] = 'LMBOT1'
    bf_df.to_csv(f"/lv_local/home/niv.b/llama/TT_input/bot_followup_LMBOT1.csv", index=False)
    fd_df = pd.read_csv(f"/lv_local/home/niv.b/train_fb_ranker/output_results/archive/feature_data_FULL.csv")
    fd_df = fd_df[fd_df.username == candidate_name]
    fd_df['username'] = 'LMBOT1'
    fd_df.to_csv(f"/lv_local/home/niv.b/llama/TT_input/feature_data_LMBOT1_new.csv", index=False)
    x = 1

    param_df = param_df[param_df['No. of trees'].isin(chosen_t) & param_df['No. of leaves'].isin(chosen_l) & param_df[
        'Learning rate'].isin(chosen_lr) & (param_df.username.str.endswith("#" + metric))].reset_index(drop=True)

    assert len(chosen_t) * len(chosen_l) * len(chosen_lr) == param_df.shape[0]

    rel_val = pd.read_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/best_val_bots.csv")
    rel_val["username"] = rel_val.username.apply(lambda x: x.replace("BOT_", ""))
    rel_val = rel_val.merge(param_df, on="username", how="left", suffixes=(None, "_y")).dropna().reset_index(
        drop=True).sort_values("true_rank_promotion_scaled", ascending=False)

    rel_test = pd.read_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/best_test_bots.csv").sort_values(
        "true_rank_promotion_scaled", ascending=False)
    rel_test["username"] = rel_test.username.apply(lambda x: x.replace("BOT_", ""))
    rel_test = rel_test.merge(param_df, on="username", how="left", suffixes=(None, "_y")).dropna().reset_index(
        drop=True)

    name_dict = dict()
    for idx, row in tqdm(rel_val.iterrows()):
        source_path = f"/lv_local/home/niv.b/train_fb_ranker/trained_models/harmonic1_VALmodel_{row.username}"
        destination_path = f"/lv_local/home/niv.b/train_fb_ranker/trained_models/chosen_bots/harmonic1_TESTmodel_LMBOT{idx + 1}"
        name_dict[row.username] = f"harmonic1_TESTmodel_LMBOT{idx + 1}"
        try:
            shutil.copy(source_path, destination_path)
        except Exception as e:
            result = f"An error occurred: {e}"

    rel_test["BOT_NAME"] = rel_test.username.apply(lambda x: name_dict[x])
    rel_val["BOT_NAME"] = rel_val.username.apply(lambda x: name_dict[x])
    rel_test.drop(['username', 'metric', 'No. of threshold candidates', 'Stop early'], axis=1).to_csv(
        "/lv_local/home/niv.b/train_fb_ranker/output_results/best_test_bots_chosen.csv"
        , index=False)
    rel_val.drop(['username', 'metric', 'No. of threshold candidates', 'Stop early'], axis=1).to_csv(
        "/lv_local/home/niv.b/train_fb_ranker/output_results/best_val_bots_chosen.csv"
        , index=False)



def create_df_and_save(rows):
    rows_df = pd.DataFrame(rows).sort_values(['min_qua', 'total_params', 'variance'], ascending=[False, False, True])
    rows_df.to_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/params_to_keep.csv", index=False)


if __name__ == '__main__':
    #### given known example  change input####
    candidate_name = 'LM384#DCG@1'
    metric = candidate_name.split("#")[1]
    print(f"############### {candidate_name} ###############\n")
    val_df, test_df, param_df = create_val_test_param_dfs()
    ##########################################

    #####only save chosen bots####
    # TODO: change to False if you want to run the whole process, change iloc to chosen index
    save_ = False
    if save_:
        chosen_dict = pd.read_csv("/lv_local/home/niv.b/train_fb_ranker/output_results/params_to_keep.csv").iloc[
            0].to_dict()
        chosen_t = add_candidate_value(param_df, eval(chosen_dict['T']), 'No. of trees', candidate_name)
        chosen_l = add_candidate_value(param_df, eval(chosen_dict['L']), 'No. of leaves', candidate_name)
        chosen_lr = add_candidate_value(param_df, eval(chosen_dict['LR']), 'Learning rate', candidate_name)
        save_chosen(param_df, chosen_t, chosen_l, chosen_lr, metric,candidate_name)
        exit()
    ##############################

    all_combinations, rel_val = calculate_combinations(val_df, param_df, candidate_name, metric)
    # unique_T_set = set(rel_val['No. of trees'].unique())
    # unique_T = list(unique_T_set.union({param_df[param_df.username == candidate_name]['No. of trees'].values[0]}))
    #
    # unique_L_set = set(rel_val['No. of leaves'].unique())
    # unique_L = list(unique_L_set.union({param_df[param_df.username == candidate_name]['No. of leaves'].values[0]}))
    #
    # unique_LR_set = set(rel_val['Learning rate'].unique())
    # unique_LR = list(unique_LR_set.union({param_df[param_df.username == candidate_name]['Learning rate'].values[0]}))

    unique_T = val_df['No. of trees'].unique()
    unique_L = val_df['No. of leaves'].unique()
    unique_LR = val_df['Learning rate'].unique()

    rows = []
    min_qua = 0
    total_params = 0
    variance = 0
    for comb in tqdm(all_combinations, total=len(all_combinations), desc="Processing Combinations"):
        row = process_combination(comb, rel_val, unique_T, unique_L, unique_LR)
        if row is not None:
            rows.append(row)
            if row['min_qua'] > min_qua:
                min_qua = row['min_qua']
                create_df_and_save(rows)
                print(f'Number of combinations: {len(rows)}')
                print('Improved on min_qua ', row)
            elif row['total_params'] > total_params:
                total_params = row['total_params']
                create_df_and_save(rows)
                print(f'Number of combinations: {len(rows)}')
                print('Improved on total_params ', row)
