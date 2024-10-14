from datasets import load_dataset
import json
from tqdm import tqdm


# Load the dataset
base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
num_shards = 46  # Number of webdataset tar files



def download_data(base_url, num_shards):
    # Download the data
    print("Downloading data...")
    urls = [base_url.format(i=i) for i in range(num_shards)]
    dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    return dataset
    


def extract_prompts(dataset, jsonl_file_path):
    # Write data to the jsonl file
    prompts = {}
    print('Extracting data to:', jsonl_file_path)

    with open(jsonl_file_path, 'w') as f:
        with tqdm(desc="Processing prompts", unit=" prompt") as pbar:
            for index, row in enumerate(dataset):
                prompts[index] = row['json']['prompt']
                f.write(json.dumps(prompts[index]) + '\n')
                
                pbar.update(1)


def read_data(jsonl_file_path):
    # Read data from the jsonl file
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            print(row)

def load_prompts_from_jsonl(file_path):
    print('Loading prompts from:', file_path)
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line) 
            prompts.append(data)  
    print("Data loaded successfully.")
    return prompts


if __name__ == "__main__":
    jsonl_file_path = r"C:\Users\jov2bg\Desktop\PromptSearch\search_engine\data\prompts_data_new.jsonl"
    num_shards = 1
    dataset = download_data(base_url, num_shards)
    extract_prompts(dataset, jsonl_file_path)
    read_data(jsonl_file_path)