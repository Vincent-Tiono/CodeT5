from datasets import DatasetDict, concatenate_datasets, Dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", nargs="+", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()


    datasets = [DatasetDict.load_from_disk(dataset_dict_path) for dataset_dict_path in args.dataset_dir]
    train_dataset = concatenate_datasets([ds["train"] for ds in datasets])
    val_dataset = concatenate_datasets([ds["val"] for ds in datasets])
    test_dataset = concatenate_datasets([ds["test"] for ds in datasets])
    merged_dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })
    merged_dataset.save_to_disk(args.output_dir)


