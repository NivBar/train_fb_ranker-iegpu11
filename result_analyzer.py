import shutil

import pandas as pd
import os
from tqdm import tqdm


def create_res_table():
    metrics = ["NDCG@1", "DCG@1", "RR@1", "ERR@1"]
    res_dir_base = "/lv_local/home/niv.b/train_fb_ranker/output_results/"

    rows = []
    for metric in tqdm(metrics):
        for file in os.listdir(res_dir_base + metric):
            if file.endswith(".txt"):
                with open(res_dir_base + metric + "/" + file, "r") as f:
                    lines = f.readlines()
                    # DCG@1   all   5.211154665140297
                    test_metric, _, score = lines[-1].split()
                    # MART2_NDCG@1.scaled_rank_promotion.DCG@1.txt
                    model, test_measure, _ = file.split(".", 2)
                    rows.append(
                        {"model": model, "test_measure": test_measure, "test_metric": test_metric, "score": score})

    df = pd.DataFrame(rows)
    df.to_csv("measure_results.csv", index=False)


if __name__ == '__main__':
    create_res_table()
    df = pd.read_csv("measure_results.csv")

    val_paths = ["./test_files/rank_test.txt", "./test_files/rank_promotion_test.txt",
                 "./test_files/scaled_rank_promotion_test.txt"]

    models = []
    index = 1
    for tree in [800, 1000, 1200, 1400, 1600]:
        for leaf in [6, 10, 14, 18, 22]:
            for shrinkage in [0.01, 0.015, 0.02, 0.025, 0.03]:
                for v in val_paths:
                    models.append(
                        {"LMmodel": f"LM{index}", "tree": tree, "leaf": leaf, "shrinkage": shrinkage, "val_path": v})
                    index += 1
    model_df = pd.DataFrame(models)
    df["LMmodel"] = df["model"].apply(lambda x: x.split("#")[0])
    df = df.merge(model_df, on="LMmodel", how='left')
    for col in ["tree", "leaf", "shrinkage"]:
        print(f"unique values for {col}: {df[col].unique()}")

    gb_df = df.groupby(["test_measure", "test_metric"])
    for name, group in gb_df:
        print(name)
        max_score = group.score.max()
        print(f"Max score: {max_score}")
        corr_values = group[['score', 'tree', 'leaf', 'shrinkage']].corr()['score']
        filtered_corr = corr_values[abs(corr_values) > 0.2]
        filtered_corr = filtered_corr[filtered_corr.index != 'score']  # Exclude self-correlation
        print(filtered_corr if not filtered_corr.empty else "No strong correlations")
        print('\n')

    tops = dict()
    # k = 20
    # for i in range(1, k+1):

    i = 1
    while len(tops) < 30:
        print("################## " + str(i) + " ##################")
        gb_df_max = df.groupby(["test_measure", "test_metric"]).apply(
            lambda x: x.nlargest(i, 'score', keep='all')).sort_values('score', ascending=False)

        print(gb_df_max.model.value_counts().sort_values(ascending=False).head(3) / 12)
        temp_df = gb_df_max.model.value_counts().sort_values(ascending=False).reset_index()
        temp_df = temp_df[temp_df["count"] == temp_df["count"].max()]
        for idx, row in temp_df.iterrows():
            if row["model"] not in tops:
                tops[row["model"]] = []
            tops[row["model"]].append(i)
        i += 1
        print(f"tops no.: {len(tops)}")
    tops = sorted(tops.items(), key=lambda x: len(x[1]), reverse=True)
    print("\n".join([str(x) for x in tops]))

    for model,_ in tops:
        shutil.copy2("/lv_local/home/niv.b/train_fb_ranker/trained_models/harmonic1_model_" + model, "/lv_local/home/niv.b/train_fb_ranker/trained_models/best_models/")
    x = 1
