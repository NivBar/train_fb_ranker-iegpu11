import os
import concurrent.futures
import logging
import subprocess
import random

from tqdm import tqdm

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_command(command):
    """
    Function to run a bash command.
    """
    try:
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        return stdout.decode('utf-8')
    except Exception as e:
        logging.error(f"Error running command '{command}': {e}")
        return None


def run_commands(commands):
    if not commands:
        logging.info("No commands to process.")
        return

    # Using ThreadPoolExecutor to run subprocesses in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 10) as executor:
        # Future to command mapping
        future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}
        for future in tqdm(concurrent.futures.as_completed(future_to_command), total=len(future_to_command), miniters=100):
            command = future_to_command[future]
            try:
                result = future.result()
                # logging.info(f"Command '{command}' executed with output: {result}")
            except Exception as e:
                print(f"Command '{command}' generated an exception: {e}")
                continue
                # logging.error(f"Command '{command}' generated an exception: {e}")


if __name__ == "__main__":
    is_train = False

    if is_train:
        #### Training ####
        file_path = '/lv_local/home/niv.b/train_fb_ranker/harmonic_features_1'

        command_base = f"/lv_local/home/niv.b/jdk-21.0.1/bin/java -jar RankLib-2.18.jar -train {file_path}"

        model_dir = "/lv_local/home/niv.b/train_fb_ranker/trained_models"
        prefix = "harmonic1_model_"

        # comm_dir = {
        #     "LNET1": f"{command_base} -ranker 7 -save {model_dir}/{prefix}LNET1",
        #     # Standard ListNet model using default settings.
        #     "RNET1": f"{command_base} -ranker 1 -epoch 100 -layer 1 -node 10 -lr 0.00005 -save {model_dir}/{prefix}RNET1",
        # }

        #  "LMART9": f"{command_base} -ranker 6 -tree 1100 -leaf 22 -shrinkage 0.09 -tc 64 -mls 9 -estop 50 -metric2t NDCG@1 -save {model_dir}/{prefix}LMART9",
        val_paths = ["./test_files/rank_test.txt", "./test_files/rank_promotion_test.txt",
                      "./test_files/scaled_rank_promotion_test.txt"]

        comm_dir = {}
        index = 1
        for tree in [800, 1000, 1200, 1400, 1600]:
            for leaf in [6, 10, 14, 18, 22]:
                for shrinkage in [0.01, 0.015, 0.02, 0.025, 0.03]:
                    for v in val_paths:
                        comm_dir[
                            f"LM{index}"] = f"{command_base} -validate {v} -ranker 6 -tree {tree} -leaf {leaf} -shrinkage {shrinkage} " \
                                            f"-metric2t TRAIN_METRIC -save SAVE_PATH"
                        index += 1

        already_created = [x.split("_")[-1] for x in os.listdir('/lv_local/home/niv.b/train_fb_ranker/trained_models/')
                           if
                           "harmonic1_model" in x]

        train_commands = dict()

        metrics = ["NDCG@1", "DCG@1", "RR@1", "ERR@1"]
        metrics = sorted(metrics)

        for k, v in comm_dir.items():
            if k in already_created:
                continue
            for metric in metrics:
                train_commands[(k, metric)] = v.replace("TRAIN_METRIC", metric).replace("SAVE_PATH",
                                                                                        f"{model_dir}/{prefix}{k}#{metric}")
        x = 1
        run_commands(list(train_commands.values()))

    else:
        ##### Testing #####
        metrics = ["NDCG@1", "DCG@1", "RR@1", "ERR@1"]
        test_paths = ["./test_files/rank_test.txt", "./test_files/rank_promotion_test.txt",
                      "./test_files/scaled_rank_promotion_test.txt"]
        test_commands = dict()
        trained_models = [file for file in os.listdir('./trained_models') if "harmonic" in file]
        for metric in metrics:
            if not os.path.exists(f"./output_results/{metric}"):
                os.makedirs(f"./output_results/{metric}")
            print(f"metric: {metric}:\n")
            for model in trained_models:
                nick = model.split("_", 2)[-1]
                for test_path in test_paths:
                    if os.path.exists(
                            f"./output_results/{metric}/{nick}.{test_path.split('./test_files/')[-1].replace('_test.txt', '')}.{metric}.txt"):
                        continue
                    test_command = f"/lv_local/home/niv.b/jdk-21.0.1/bin/java -jar RankLib-2.18.jar -load ./trained_models/{model} -test {test_path} -metric2T {metric} -idv ./output_results/{metric}/{nick}.{test_path.split('./test_files/')[-1].replace('_test.txt', '')}.{metric}.txt"
                    test_commands[(nick, metric, test_path)] = test_command

        comm_list = list(test_commands.values())
        assert len(comm_list) == len(trained_models) * len(metrics) * len(test_paths)
        random.shuffle(comm_list)
        run_commands(comm_list)
