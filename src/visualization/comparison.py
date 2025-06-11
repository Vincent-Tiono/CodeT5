import argparse
import matplotlib.pyplot as plt
import json
import os
import ipdb
import re
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Plot the results of the experiment')
    parser.add_argument('--dir', type=str, help='The root directory of experiments', nargs='+')
    parser.add_argument('--baseline_dir', type=str, help='The root directory of baseline experiments', nargs='+')
    parser.add_argument('--output', type=str, help='The output directory')
    parser.add_argument('--length', type=int, default=1)
    args = parser.parse_args()
    return args

def parse_epoch(root):
    pattern = r'epoch-\d+'
    epoch = re.findall(pattern, root)[0]
    return int(epoch.split('-')[-1])

def get_data(dir):
    data = {}
    current_epoch = 0
    for root, dirs, files in os.walk(dir):
        if len(dirs) == 0:
            epoch = parse_epoch(root)
            if epoch > current_epoch:
                current_epoch = epoch
                for file in files:
                    if file.endswith('.json') and not "beam" in file:
                        with open(os.path.join(root, file), 'r') as f:
                            data[file.replace("_result.json", "")] = json.load(f)
    return data

def plot(data, output, x_label, exp_name="", length=1):
    """
    Plot the results of the experiment. 
    Args:
        data (list): A list of data dicts.
        output (str): The output directory.
    """
    os.makedirs(output, exist_ok=True)

    def plot_by_key(key):
        # plot the specific key
        train_exact = [d["train"][key] for d in data]
        eval_exact = [d["eval"][key] for d in data]
        test_exact = [d["test"][key] for d in data]
        plt.plot(np.arange(len(data)) + length, train_exact, label="train")
        plt.plot(np.arange(len(data)) + length, eval_exact, label="eval")
        plt.plot(np.arange(len(data)) + length, test_exact, label="test")
        plt.legend()
        plt.xlabel(x_label)
        plt.title(f"{exp_name} {key}")
        output_file = os.path.join(output, f"{key}.png")
        plt.savefig(output_file)
        plt.cla()
        plt.clf()

    # plot exact, sample and score
    plot_by_key("exact")
    plot_by_key("sample")
    plot_by_key("score")


def plot_difference(data, baseline_data, output, x_label, length):
    """
    Plot the difference between the results of the experiment and the baseline experiment. 
    Args:
        data (list): A list of data dicts.
        baseline_data (list): A list of data dicts of the baseline
        output (str): The output directory.
    """
    os.makedirs(output, exist_ok=True)

    def plot_by_key(key: str):
        # plot the specific key
        train_exact = np.array([d["train"][key] for d in data])
        eval_exact = np.array([d["eval"][key] for d in data])
        test_exact = np.array([d["test"][key] for d in data])
        train_baseline = np.array([d["train"][key] for d in baseline_data])
        eval_baseline = np.array([d["eval"][key] for d in baseline_data])
        test_baseline = np.array([d["test"][key] for d in baseline_data])
        diff_train = (train_exact - train_baseline)
        diff_eval = (eval_exact - eval_baseline)
        diff_test = (test_exact - test_baseline)
        plt.plot(np.arange(len(data)) + length, diff_train, label="train")
        plt.plot(np.arange(len(data)) + length, diff_eval, label="eval")
        plt.plot(np.arange(len(data)) + length, diff_test, label="test")
        plt.legend()
        plt.xlabel(x_label)
        plt.title(f"{key.capitalize()} gain")
        output_file = os.path.join(output, f"{key}.png")
        plt.savefig(output_file)
        plt.cla()
        plt.clf()

    # plot exact, sample and score
    plot_by_key("exact")
    plot_by_key("sample")
    plot_by_key("score")

def plot_alias(data, baseline_data, output, x_label, length):
    """
    Plot the aliased 
    Args:
        data (list): A list of data dicts.
        baseline_data (list): A list of data dicts of the baseline
        output (str): The output directory.
    """
    os.makedirs(output, exist_ok=True)

    def get_alias(data):
        train_sample = np.array([d["train"]["sample"] for d in data])
        eval_sample = np.array([d["eval"]["sample"] for d in data])
        test_sample = np.array([d["test"]["sample"] for d in data])
        train_baseline = np.array([d["train"]["exact"] for d in data])
        eval_baseline = np.array([d["eval"]["exact"] for d in data])
        test_baseline = np.array([d["test"]["exact"] for d in data])
        diff_train = (train_sample - train_baseline)
        diff_eval = (eval_sample - eval_baseline)
        diff_test = (test_sample - test_baseline)
        return diff_train, diff_eval, diff_test

    def plot_by_data(data, title):
        diff_train, diff_eval, diff_test = get_alias(data)
        plt.plot(np.arange(len(data)) + length, diff_train, label="train")
        plt.plot(np.arange(len(data)) + length, diff_eval, label="eval")
        plt.plot(np.arange(len(data)) + length, diff_test, label="test")
        plt.legend()
        plt.xlabel(x_label)
        plt.title(f"{title.capitalize()} Alias")
        output_file = os.path.join(output, f"{title}.png")
        plt.savefig(output_file)
        plt.cla()
        plt.clf()

    def plot_alias_gain(title="alias"):
        alias_train_base, alias_eval_base, alias_test_base = get_alias(baseline_data)
        alias_train, alias_eval, alias_test = get_alias(data)
        diff_train = (alias_train - alias_train_base)
        diff_eval = (alias_eval - alias_eval_base)
        diff_test = (alias_test - alias_test_base)
        plt.plot(np.arange(len(data)) + length, diff_train, label="train")
        plt.plot(np.arange(len(data)) + length, diff_eval, label="eval")
        plt.plot(np.arange(len(data)) + length, diff_test, label="test")
        plt.legend()
        plt.xlabel(x_label)
        plt.title(f"{title.capitalize()} Gain")
        output_file = os.path.join(output, f"{title}.png")
        plt.savefig(output_file)
        plt.cla()
        plt.clf()

    # plot exact, sample and score
    plot_by_data(baseline_data, "seq2seq")
    plot_by_data(data, "clip")
    plot_alias_gain()

if __name__  == '__main__':
    args = parse_args()
    data = [get_data(dir) for dir in args.dir]
    baseline_data = [get_data(dir) for dir in args.baseline_dir]
    plot_alias(data, baseline_data, os.path.join(args.output, "alias"), "length", length=args.length)
    # plot(baseline_data, os.path.join(args.output, "baseline"), "length", "Seq2seq", length=args.length)
    # plot(data, os.path.join(args.output, "data"), "length", "CLIP+CodeT5", length=args.length)

