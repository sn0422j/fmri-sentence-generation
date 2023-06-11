import argparse
from configparser import ConfigParser
from datetime import datetime
from itertools import product
from subprocess import run
from typing import Final

import requests
from tqdm import tqdm
from yaml import safe_load

config_ini = ConfigParser()
config_ini.read("config.ini", encoding="utf-8")
line_notify_token: Final[str] = config_ini.get("LINE", "token")


def send_message(message="notification"):
    line_notify_api = "https://notify-api.line.me/api/notify"

    headers = {"Authorization": f"Bearer {line_notify_token}"}
    files = {"message": (None, message)}
    response = requests.post(line_notify_api, headers=headers, files=files)
    print("Posted!", response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Call it when you want to run commands.", action="store_true")
    parser.add_argument("--debug", help="debug mode.", action="store_true")
    args = parser.parse_args()

    if args.run:
        log_file = "./batch/log/run_{0:%Y%m%d_%H%M%S}.txt".format(datetime.now())
    else:
        log_file = "./batch/log/chk_{0:%Y%m%d_%H%M%S}.txt".format(datetime.now())

    with open("./batch/parameter.yaml") as file:
        parameter = safe_load(file)

    with open(log_file, mode="a") as f:
        f.write(f"[args] { args } \n")
        f.flush()
        f.write(f"[parameter] { parameter } \n")

    command_list = []
    # Command List Start #
    exp_id = 2
    for sub_id in parameter["sub_id_for_exp_2"]:
        command = f"python scripts/step1_preprocessing/make_dataset_pereira2018_new.py {exp_id} {sub_id}"
        # command_list.append(command)
    exp_id = 3
    for sub_id in parameter["sub_id_for_exp_3"]:
        command = f"python scripts/step1_preprocessing/make_dataset_pereira2018_new.py {exp_id} {sub_id}"
        # command_list.append(command)

    for model in parameter["model"]:
        exp_id = 2
        for sub_id in parameter["sub_id_for_exp_2"]:
            command = f"python scripts/step2_training/run_generate.py {exp_id} {sub_id} {model}"
            if args.debug:
                command += " --debug"
            # command_list.append(command)
        for sub_id, feature_selection, regressor in product(
            parameter["sub_id_for_exp_2"], parameter["feature_selection"], parameter["regressor"]
        ):
            command = f"python scripts/step2_training/run_predict.py {exp_id} {sub_id} {model} \
                --feature_selection={feature_selection} --regressor={regressor}"
            if args.debug:
                command += " --debug"
            # command_list.append(command)
            command = f"python scripts/step2_training/run_predict.py {exp_id} {sub_id} {model} \
                --feature_selection={feature_selection} --regressor={regressor} --permute"
            if args.debug:
                command += " --debug"
            # command_list.append(command)

        exp_id = 3
        for sub_id in parameter["sub_id_for_exp_3"]:
            command = f"python scripts/step2_training/run_generate.py {exp_id} {sub_id} {model}"
            if args.debug:
                command += " --debug"
            command_list.append(command)
        for sub_id, feature_selection, regressor in product(
            parameter["sub_id_for_exp_3"], parameter["feature_selection"], parameter["regressor"]
        ):
            command = f"python scripts/step2_training/run_predict.py {exp_id} {sub_id} {model} \
                --feature_selection={feature_selection} --regressor={regressor}"
            if args.debug:
                command += " --debug"
            command_list.append(command)
            command = f"python scripts/step2_training/run_predict.py {exp_id} {sub_id} {model} \
                --feature_selection={feature_selection} --regressor={regressor} --permute"
            if args.debug:
                command += " --debug"
            command_list.append(command)

        command = f"python scripts/step3_evaluation/evaluate.py {model}"
        command_list.append(command)
        command = f"python scripts/step3_evaluation/evaluate_topic.py {model}"
        command_list.append(command)
        command = f"python scripts/step3_evaluation/evaluate_topic.py {model} --permute"
        command_list.append(command)
    # Command List End #

    with open(log_file, mode="a") as f:
        for command in tqdm(command_list):
            time = "{0:%Y/%m/%d %H:%M:%S}".format(datetime.now())  # 2020/01/01 00:00:00
            print(f"[command] {time} - {command} \n")
            f.write(f"[command] {time} - {command} \n")
            f.flush()
            if args.run:
                _ = run(command, stdout=f, shell=True)  # Completed Process Returened
                send_message(command)
    if args.run:
        send_message("All process finished.")


if __name__ == "__main__":
    main()
