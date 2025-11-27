import json
import csv

def concatenate_json_to_csv(json_file, csv_file):
    # Read the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Prepare the data for CSV
    prompts = [item.get('prompt', '') for item in data]
    canonical_solutions = [item.get('canonical_solution', '') for item in data]

    # Write to CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for prompt, canonical_solution in zip(prompts, canonical_solutions):
            writer.writerow([prompt, canonical_solution])

# Example usage
# concatenate_json_to_csv('input.json', 'output.csv')
concatenate_json_to_csv("/data/home/zhangsj/Data/HumanEval/human-eval-v2-20210705.jsonl","/data/home/zhangsj/Data/HumanEval/new_columns_train.csv")