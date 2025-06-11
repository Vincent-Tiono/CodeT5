import os
import re
import argparse
import json
import pandas as pd

def parse_epoch(root):
    pattern = r'epoch-\d+'
    epoch = re.findall(pattern, root)[0]
    return int(epoch.split('-')[-1])

def get_data(dir):
    data = {}
    try:
        current_epoch = 0
        for root, dirs, files in os.walk(dir):
            if len(dirs) == 0:
                epoch = parse_epoch(os.path.split(root)[1])
                if epoch > current_epoch:
                    current_epoch = epoch
                    for file in files:
                        if file.endswith('.json') and not "beam" in file:
                            with open(os.path.join(root, file), 'r') as f:
                                data[file.replace("_result.json", "")] = json.load(f)
    except:
        print(f"Error in {dir}")
    return data

def main(args):
    
    data = [get_data(os.path.join(args.root_dir, dir)) for dir in args.dir]
    df = None
    if os.path.exists(args.output_file):
        df = pd.read_csv(args.output_file)
    
    experiments = [os.path.split(dir)[1] for dir in args.dir]
    columns = {"exp": experiments}
    for metric in ["score", "exact", "sample"]:
        for set_name in ["train", "eval", "test"]:
            col = []
            for d in data:
                if set_name in d:
                    col.append(d[set_name][metric])
                else:
                    col.append(None)
            # col = [d[set_name][metric] for d in data]
            columns[f"{metric}-{set_name}"] = col
    if df is not None:
        df = pd.concat([df, pd.DataFrame(columns)])
    else:
        df = pd.DataFrame(columns)
    df.to_csv(args.output_file, index=False, float_format='%.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result of the experiments')
    parser.add_argument('--dir', type=str, help='The root directory of experiments', nargs='+')
    parser.add_argument('--root_dir', type=str, help='The root directory of experiment', default="./")
    parser.add_argument('--output_file', type=str, help='The output file')
    args = parser.parse_args()
    main(args)
