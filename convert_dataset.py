import h5py
import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
import ipdb
from typing import List
from tqdm.auto import tqdm
from src.development.hprl_function_dsl import DSLProb_option_v2
from src.development.hprl_function_karel import HPRLFunctionCall, function_table, branch_table

import os
import argparse
import numpy as np
import shutil

from hprl_karel_env.dsl import get_DSL_option_v2

def convert_io_dataset(args):
    dataset_dir = args.dataset_dir
    all_files = os.listdir(dataset_dir)
    hdf5_files =  [f for f in all_files if f.endswith('.hdf5')]

    def get_ids(hdf5_file: str):
        id_file = hdf5_file.replace("data", "id").replace(".hdf5", ".txt")
        with open(os.path.join(dataset_dir, id_file)) as f:
            ids = [line.split(" ")[0] for line in f.read().splitlines()]
        return ids

    def get_program_id(id: str):
        return int(id.split("_")[1])

    def get_file_id(ids: List[str]):
        return min([get_program_id(id) for id in ids])

    sorted_hdf5_files = sorted(hdf5_files, key=lambda x: get_file_id(get_ids(x)))

    inputs = []
    outputs = []
    programs = []
    num_train = args.num_train
    num_test = args.num_test
    num_val = args.num_val
    num_total = num_train + num_test + num_val

    progress_bar = tqdm(range(num_total), desc="Loading dataset")

    dsl = get_DSL_option_v2(seed=123)
    for f in sorted_hdf5_files:
        ids = get_ids(f)
        with h5py.File(os.path.join(dataset_dir, f), 'r') as hdf5_file:
            for id in ids:
                # datum contains a_h, a_h_len, program, s_h, s_h_len
                datum = hdf5_file[id]
                # ipdb.set_trace()
                programs.append(dsl.intseq2str(datum["program"]))
                s_h = np.array(datum["s_h"])
                s_h_len = np.array(datum["s_h_len"])
                input = np.stack([s_h[i, 0] for i in range(s_h.shape[0])])
                inputs.append(input)
                output = np.stack([s_h[i, l-1] for i, l in enumerate(s_h_len)])
                outputs.append(output)
                # both inputs and outpus in shape (num_demo, h, w, c)
                progress_bar.update(1)

    # split sets


    inputs_train = inputs[:num_train]
    outputs_train = outputs[:num_train]
    programs_train = programs[:num_train]

    inputs_test = inputs[num_train:num_train+num_test]
    outputs_test = outputs[num_train:num_train+num_test]
    programs_test = programs[num_train:num_train+num_test]

    inputs_val = inputs[num_train+num_test:num_train+num_test+num_val]
    outputs_val = outputs[num_train+num_test:num_train+num_test+num_val]
    programs_val = programs[num_train+num_test:num_train+num_test+num_val]

    # save
    train_dataset = Dataset.from_dict({
        "inputs": inputs_train,
        "outputs": outputs_train,
        "program": programs_train
    })
    val_dataset = Dataset.from_dict({
        "inputs": inputs_val,
        "outputs": outputs_val,
        "program": programs_val
    })
    test_dataset = Dataset.from_dict({
        "inputs": inputs_test,
        "outputs": outputs_test,
        "program": programs_test
    })
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })
    dataset.save_to_disk(args.output_dir)

def convert_demo_dataset(args):
    dataset_dir = args.dataset_dir
    os.makedirs(dataset_dir, exist_ok=True)
    all_files = os.listdir(dataset_dir)
    hdf5_files =  [f for f in all_files if f.endswith('.hdf5')]

    def get_ids(hdf5_file: str):
        id_file = hdf5_file.replace("data", "id").replace(".hdf5", ".txt")
        with open(os.path.join(dataset_dir, id_file)) as f:
            ids = [line.split(" ")[0] for line in f.read().splitlines()]
        return ids

    def get_program_id(id: str):
        return int(id.split("_")[1])

    def get_file_id(ids: List[str]):
        return min([get_program_id(id) for id in ids])

    sorted_hdf5_files = sorted(hdf5_files, key=lambda x: get_file_id(get_ids(x)))


    num_train = args.num_train
    num_test = args.num_test
    num_val = args.num_val
    num_total = num_train + num_test + num_val

    progress_bar = tqdm(range(num_total), desc="Loading dataset")

    dsl = get_DSL_option_v2(seed=123)
    for i, f in enumerate(sorted_hdf5_files):
        ids = get_ids(f)
        with h5py.File(os.path.join(dataset_dir, f), 'r') as hdf5_file:
            inputs = []
            inputs_length = []
            programs = []

            for id in ids:
                # datum contains a_h, a_h_len, program, s_h, s_h_len
                datum = hdf5_file[id]
                # ipdb.set_trace()
                programs.append(dsl.intseq2str(datum["program"]))
                # ipdb.set_trace()
                inputs.append(datum["s_h"][:])
                inputs_length.append(datum["s_h_len"][:])
                progress_bar.update(1)

            train_dataset = Dataset.from_dict({
                "inputs": inputs,
                "inputs_length": inputs_length,
                "program": programs
            })
            train_dataset.save_to_disk(os.path.join(args.output_dir, f"{i:03d}_split"))

    whole_datasets = []

    for i in range(len(sorted_hdf5_files)):
        train_dataset = Dataset.load_from_disk(os.path.join(args.output_dir, f"{i:03d}_split"))
        whole_datasets.append(train_dataset)
    whole_datasets = concatenate_datasets(whole_datasets)
    datasets = whole_datasets.train_test_split(test_size=num_test+num_val, shuffle=False)
    train_dataset = datasets["train"]
    val_test_dataset = datasets["test"].train_test_split(test_size=num_val, shuffle=False)
    val_dataset = val_test_dataset["train"]
    test_dataset = val_test_dataset["test"]
    # ipdb.set_trace()
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })
    dataset.save_to_disk(args.output_dir)

    for i in range(len(sorted_hdf5_files)):
        # remove the directory
        shutil.rmtree(os.path.join(args.output_dir, f"{i:03d}_split"))
    
''''''
# Collect state history, perception history, action history, function call history, branch history
def convert_function_dataset(args):
    dataset_dir = args.dataset_dir
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = os.listdir(dataset_dir)
    hdf5_files =  [f for f in all_files if f.endswith('.hdf5')]

    def get_ids(hdf5_file: str):
        id_file = hdf5_file.replace("data", "id").replace(".hdf5", ".txt")
        with open(os.path.join(dataset_dir, id_file)) as f:
            ids = [line.split(" ")[0] for line in f.read().splitlines()]
        return ids

    def get_program_id(id: str):
        return int(id.split("_")[1])

    def get_file_id(ids: List[str]):
        return min([get_program_id(id) for id in ids])

    ''''''
    def inspect_program_execution(dsl, karel, code, state):
        print("\nDETAILED PROGRAM INSPECTION")
        print(f"Program: {code}")
        
        # Reset Karel
        karel.set_new_state(state)
        
        # Enable step-by-step tracing if the class supports it
        if hasattr(karel, 'enable_tracing'):
            karel.enable_tracing = True
        
        # Run with tracing
        print("Starting step-by-step execution:")
        dsl.run(karel, code)
        
        print(f"Final perception history length: {len(karel.p_v_h)}")
    ''''''

    sorted_hdf5_files = sorted(hdf5_files, key=lambda x: get_file_id(get_ids(x)))


    num_train = args.num_train
    num_test = args.num_test
    num_val = args.num_val
    num_total = num_train + num_test + num_val

    progress_bar = tqdm(range(num_total), desc="Loading dataset")


    dsl = DSLProb_option_v2(seed=123)
    karel = HPRLFunctionCall()
    perception_list = karel.get_perception_list()
    num_perception = len(perception_list)

    dsl = get_DSL_option_v2(seed=123)
    for fid, f in enumerate(sorted_hdf5_files):
        ids = get_ids(f)
        with h5py.File(os.path.join(dataset_dir, f), 'r') as hdf5_file:
            inputs = []
            inputs_length = []
            programs = []
            perceptions = []
            perception_length = []
            actions = []
            action_length = []
            functions = []
            branches = []

            for id in ids:
                # datum contains a_h, a_h_len, program, s_h, s_h_len
                datum = hdf5_file[id]
                
            
                code = dsl.intseq2str(datum["program"]) # -- actual program
                # print(datum["program"])
                # print(code)
                # print("\n")
                empty_perception_count = 0
                stacked_perception_count = 0
                
                s_h = datum["s_h"][:] # -- state history
                s_h_len = datum["s_h_len"][:] # -- state history length
                num_demo = s_h.shape[0]

                # process perception
                p_h = []
                p_h_len = []
                f_v_h = []
                b_v_h = []
                
                problematic_demos = []
                for j in range(num_demo):
                    '''
                    # This is the original code
                    karel.set_new_state(s_h[j, 0])
                    dsl.run(karel, code)
                    '''
                    
                    karel.set_new_state(s_h[j, 0])
                    
                    # print("inital state:", s_h[j, 0])
                    # # Print info about initial state
                    # print(f"  Initial state summary: {karel.get_state_summary() if hasattr(karel, 'get_state_summary') else 'N/A'}")
                    
                    # Add debug before running
                    print(f"  Running program: {code}")
                    initial_p_v_h_len = len(karel.p_v_h) if hasattr(karel, 'p_v_h') else 0
                    print(f"  Initial p_v_h length: {initial_p_v_h_len}")
                    
                    # Run program
                    exec_history = dsl.run(karel, code)
                    # print(exec_history)
                    
                    # Check perception history length
                    final_p_v_h_len = len(karel.p_v_h) if hasattr(karel, 'p_v_h') else 0
                    print(f"  Final p_v_h length: {final_p_v_h_len}")
                    
                    if len(karel.p_v_h) > 0:
                        p_v_h = np.stack(karel.p_v_h)
                        stacked_perception_count += 1
                    else:
                        # print(f"Warning: Empty perception history for demo {j} in program '{id}'. Using default empty perception.")
                        # Create an empty perception vector of appropriate shape
                        problematic_demos.append((j, s_h[j, 0].copy()))
                        p_v_h = np.zeros((0, num_perception), dtype=bool)
                        empty_perception_count += 1
                    
                    p_h.append(p_v_h)
                    p_h_len.append(len(p_v_h))
                    karel.add_callee()
                    f_v_h.append(karel.function_history)
                    b_v_h.append(karel.branch_history)
                
                ''''''
                if problematic_demos:
                    print("\n==== DETAILED INSPECTION OF PROBLEMATIC DEMOS ====")
                    for demo_idx, state in problematic_demos:
                        print(f"\nInspecting problematic demo {demo_idx} in detail:")
                        inspect_program_execution(dsl, karel, code, state)
                ''''''
                    
                
                f_h = np.zeros((len(s_h), max(s_h_len), len(function_table)), dtype=bool)
                b_h = np.zeros((len(s_h), max(s_h_len), len(branch_table)), dtype=bool)
                for j in range(len(s_h)):
                    if s_h_len[j] > 0:
                        f_h[j, :s_h_len[j]] = f_v_h[j]
                        b_h[j, :s_h_len[j]] = b_v_h[j]
                
                demos_p_h = np.zeros([num_demo, np.max(p_h_len), num_perception], dtype=bool)
                for j, p in enumerate(p_h):
                    demos_p_h[j, :p.shape[0]] = p

                # process action padding, current padding using zero, but we can use -100 in favor of cross entropy loss
                a_h = datum["a_h"][:]
                a_h_len = datum["a_h_len"][:]
                for j in range(num_demo):
                    a_h[j, a_h_len[j]:] = -100

                
                programs.append(code)
                inputs.append(s_h)
                inputs_length.append(datum["s_h_len"][:])
                perceptions.append(demos_p_h)
                perception_length.append(p_h_len)
                actions.append(a_h)
                action_length.append(a_h_len)
                functions.append(f_h)
                branches.append(b_h)
                progress_bar.update(1)
            
                # After processing all examples in a file
                print(f"File {f}: Processed {stacked_perception_count + empty_perception_count} demonstrations")
                print(f"  - With actual perception history: {stacked_perception_count} ({stacked_perception_count/(empty_perception_count + stacked_perception_count)*100:.2f}%)")
                print(f"  - With empty perception history: {empty_perception_count} ({empty_perception_count/(empty_perception_count + stacked_perception_count)*100:.2f}%)")
                        
            train_dataset = Dataset.from_dict({
                "inputs": inputs,
                "inputs_length": inputs_length,
                "perceptions": perceptions,
                "perceptions_length": perception_length,
                "actions": actions,
                "actions_length": action_length,
                "functions": functions,
                "branches": branches,
                "program": programs
            })
            train_dataset.save_to_disk(os.path.join(args.output_dir, f"{fid:03d}_split"))

    whole_datasets = []

    for fid in range(len(sorted_hdf5_files)):
        train_dataset = Dataset.load_from_disk(os.path.join(args.output_dir, f"{fid:03d}_split"))
        whole_datasets.append(train_dataset)
    whole_datasets = concatenate_datasets(whole_datasets)
    datasets = whole_datasets.train_test_split(test_size=num_test+num_val, shuffle=False)
    train_dataset = datasets["train"]
    val_test_dataset = datasets["test"].train_test_split(test_size=num_val, shuffle=False)
    val_dataset = val_test_dataset["train"]
    test_dataset = val_test_dataset["test"]
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })
    dataset.save_to_disk(args.output_dir)

    for i in range(len(sorted_hdf5_files)):
        # remove the directory
        shutil.rmtree(os.path.join(args.output_dir, f"{i:03d}_split"))
''''''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets/karel_cover_branch")
    parser.add_argument("--output_dir", type=str, default="datasets/hprl_synthesis")
    parser.add_argument('--num_train', type=int, default=100000, help='num train')
    parser.add_argument('--num_test',  type=int, default=5000,  help='num test')
    parser.add_argument('--num_val',   type=int, default=5000,  help='num val')
    args = parser.parse_args()
    # convert_io_dataset(args)
    convert_demo_dataset(args)
    
    ''''''
    convert_function_dataset(args)
    ''''''
