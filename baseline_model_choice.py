import pandas as pd
from utils import run_bash_command
from tqdm import tqdm
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


def process_model(model_name, model_dir, features_file_path, df):
    predictions_file_path = f'/lv_local/home/niv.b/train_fb_ranker/output_results/ranker_test_results/predictions_{model_name}.txt'

    if not os.path.exists(predictions_file_path):
        command = f"/lv_local/home/niv.b/jdk-21.0.1/bin/java -jar RankLib-2.18.jar -load {model_dir}/{model_name} -rank {features_file_path} -score {predictions_file_path}"
        output = run_bash_command(command)

    score_column = pd.read_csv(predictions_file_path, header=None, delimiter='\t', usecols=[2])

    if int(score_column.isna().sum()) > 0:
        print(f"ERROR in {model_name}: score column contains NaN values")

    df["score"] = score_column
    df['position'] = df.groupby("orig_docno")['score'].rank(method='first', ascending=False).astype(int)
    res_dict = df[df.position == 1][['rank', 'rank_promotion', 'scaled_rank_promotion']].mean().to_dict()
    res_dict["model"] = model_name.split("_")[-1]

    return res_dict


def main(path, state):
    df = pd.read_csv(path)
    df["orig_docno"] = df.docno.apply(lambda x: x.split("$")[0])
    models = [model for model in os.listdir("/lv_local/home/niv.b/train_fb_ranker/trained_models") if
              "baseline_model" in model]

    rows = []

    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(process_model, model_name, "trained_models", "baseline_dataset_test_r4.txt",
                                           df[["docno", "orig_docno", "rank", "rank_promotion",
                                               "scaled_rank_promotion"]]): model_name for model_name in models}

        for future in tqdm(as_completed(future_to_model), total=len(models)):
            model_name = future_to_model[future]
            try:
                res_dict = future.result()
                rows.append(res_dict)
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')

    res_df = pd.DataFrame(rows)
    res_df.sort_values("scaled_rank_promotion", ascending=True).to_csv(
        f"/lv_local/home/niv.b/train_fb_ranker/output_results/ranker_test_results/baseline_model_choice_results_{state}.csv",
        index=False)


if __name__ == '__main__':
    test_paths = "baseline_dataset_test_r4_summary.csv"
    val_paths = "baseline_dataset_train_r3_validation_set.csv"
    if not os.path.exists(test_path) or not os.path.exists(val_path):
        raise Exception("Please make sure test and val paths exist")
    if not os.path.exists("/lv_local/home/niv.b/train_fb_ranker/output_results/ranker_test_results/baseline_model_choice_results_test.csv"):
        main(test_path, "test")
    if not os.path.exists("/lv_local/home/niv.b/train_fb_ranker/output_results/ranker_test_results/baseline_model_choice_results_val.csv"):
        main(val_path, "val")
