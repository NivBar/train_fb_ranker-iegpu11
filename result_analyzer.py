import shutil
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import config as conf

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_feature_correlations_with_means_medians(df, title):
    # Set up the matplotlib figure
    plt.figure(figsize=(18, 6))

    # Titles and labels
    features = ['tree', 'leaf', 'shrinkage']
    score_column = 'score'  # Adjust if your score column has a different name

    # Creating subplots for each feature
    for i, feature in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        sns.scatterplot(x=df[feature], y=df[score_column])

        # Calculate means and medians
        mean_values = df.groupby(feature)[score_column].mean()
        median_values = df.groupby(feature)[score_column].median()

        # Plot means and medians
        plt.scatter(mean_values.index, mean_values.values, color='green', label='Mean')
        plt.scatter(median_values.index, median_values.values, color='red', label='Median')

        # Connect means and medians
        plt.plot(mean_values.index, mean_values.values, color='green', linestyle='-', linewidth=2)
        plt.plot(median_values.index, median_values.values, color='red', linestyle='-', linewidth=2)

        plt.title(f'{title}: {feature}')
        plt.xlabel(feature)
        plt.ylabel(score_column)
        plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    file_name = title.replace(' ', '_') + '.png'
    plt.savefig(file_name)
    plt.show()
    plt.close()
    print(f"Plot saved as '{file_name}'")


def create_res_table():
    # metrics = ["NDCG@1", "DCG@1", "RR@1", "ERR@1"]
    metrics = ["ERR@1"]
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

    # val_paths = ["./test_files/rank_test.txt", "./test_files/rank_promotion_test.txt",
    #              "./test_files/scaled_rank_promotion_test.txt"]

    models = []
    index = 1
    for tree in conf.tree_vals:
        for leaf in conf.leaf_vals:
            for shrinkage in conf.shrinkage_vals:
                models.append(
                    {"LMmodel": f"LM{index}", "tree": tree, "leaf": leaf, "shrinkage": shrinkage})
                index += 1

    model_df = pd.DataFrame(models)
    df["LMmodel"] = df["model"].apply(lambda x: x.split("#")[0])
    df = df.merge(model_df, on="LMmodel", how='left')
    for col in ["tree", "leaf", "shrinkage"]:
        print(f"unique values for {col}: {df[col].unique()}")

    gb_df = df.groupby(["test_measure", "test_metric"])
    for name, group in gb_df:
        print(name)
        plot_feature_correlations_with_means_medians(group, str(name))
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

        print(gb_df_max)

        # print(gb_df_max.model.value_counts().sort_values(ascending=False).head(3))
        temp_df = gb_df_max.model.value_counts().sort_values(ascending=False).reset_index()
        temp_df = temp_df[temp_df["count"] == temp_df["count"].max()]
        for idx, row in temp_df.iterrows():
            if row["model"] not in tops:
                tops[row["model"]] = []
            tops[row["model"]].append(i)

        # TODO: remove this for more models
        if max_score > 0.0988: # current best score
            if not os.path.exists(f"/lv_local/home/niv.b/train_fb_ranker/best_models/"):
                os.makedirs(f"/lv_local/home/niv.b/train_fb_ranker/best_models/")
            if os.path.exists(f"/lv_local/home/niv.b/train_fb_ranker/best_models/models_df.csv"):
                df = pd.read_csv(f"/lv_local/home/niv.b/train_fb_ranker/best_models/models_df.csv")
                gb_df_max = pd.concat([gb_df_max, df]).drop_duplicates()

            gb_df_max.to_csv(
                f"/lv_local/home/niv.b/train_fb_ranker/best_models/models_df.csv", index=False)
            for model in tops.keys():
                if not os.path.exists("/lv_local/home/niv.b/train_fb_ranker/harmonic1_model_" + model):
                    shutil.copy2("/lv_local/home/niv.b/train_fb_ranker/trained_models/harmonic1_model_" + model,
                                 "/lv_local/home/niv.b/train_fb_ranker/best_models/")
        break

        i += 1
        print(f"tops no.: {len(tops)}")

    tops = sorted(tops.items(), key=lambda x: len(x[1]), reverse=True)
    print("\n".join([str(x) for x in tops]))

    # for model,_ in tops:
    #     shutil.copy2("/lv_local/home/niv.b/train_fb_ranker/trained_models/harmonic1_model_" + model, "/lv_local/home/niv.b/train_fb_ranker/trained_models/best_models/")
    x = 1
