import os
import json
import time
import sys
import argparse
import logging
from openai import OpenAI
from datetime import datetime
from dataset_templates import TemplateManager
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# --- 配置区域 ---
API_KEY = "HLwaq5OfA0dhif7NLl7afm7kUFuP0toS"  # 记得替换
BASE_URL = "https://api.deepinfra.com/v1/openai"

# 定义模型价格字典 (在此处维护价格表)
MODEL_PRICING = {
    "MiniMaxAI/MiniMax-M2": {"in": 0.27, "out": 1.15},
    "deepseek-ai/DeepSeek-V3.2-Exp": {"in": 0.27, "out": 0.40},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"in": 0.09, "out": 0.57},
    "moonshotai/Kimi-K2-Instruct-0905": {"in": 0.50, "out": 2.00},
    "openai/gpt-oss-120b": {"in": 0.05, "out": 0.24},
    "google/gemma-3-27b-it": {"in": 0.09, "out": 0.16},
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {"in": 0.075, "out": 0.20},
    "google/gemma-3-12b-it": {"in": 0.04, "out": 0.13},
    "deepseek-ai/DeepSeek-V3.1-Terminus": {"in": 0.21, "out": 0.79},
    # 在这里添加更多模型...
}

# 数据集路径映射
DATASET_MAPPING = {
    "GSM8k": "Dataset/GSM8k/train.cleand.jsonl",
    "MMLU_PRO": "Dataset/MMLU_PRO/mmlupro.jsonl",
    "GSM8k_Test": "Dataset/GSM8k/test.cleand.jsonl",
    "IFEVAL":"Dataset/IFEVAL/ifeval_input_data.jsonl",
    "BBH":"Dataset/BBH/bbh.jsonl",
    "Human_Eval":"Dataset/HumanEval/human-eval-v2-20210705.jsonl"
    # 在这里添加更多数据集映射...
}

Dataset_Instructions = {
    # 在这里添加数据集对应的特殊指令（如果有）
}

# --- 初始化日志 (打印到控制台，由 Shell 重定向到文件) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
# Suppress httpx and openai INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_completion_and_cost(prompt, model_name, prices):
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = chat_completion.choices[0].message.content
        p_tokens = chat_completion.usage.prompt_tokens
        c_tokens = chat_completion.usage.completion_tokens
        
        # 计算价格 (假设价格单位是 Per Million)
        total_cost = (p_tokens * prices["in"] / 1000000) + (c_tokens * prices["out"] / 1000000)
        return content, total_cost
    except Exception as e:
        raise e

def safe_run_single(prompt, model_name, prices):
    """重试逻辑：2s -> 30s -> Fail"""
    # 第一次重试
    try:
        return get_completion_and_cost(prompt, model_name, prices)
    except Exception as e:
        logger.warning(f"[Retry 2s] {model_name}: {str(e)[:100]}")
        time.sleep(2)
        
    # 第二次重试
    try:
        return get_completion_and_cost(prompt, model_name, prices)
    except Exception as e:
        logger.warning(f"[Retry 10s] {model_name}: {str(e)[:100]}")
        time.sleep(10)

    # 第三次重试
    try:
        return get_completion_and_cost(prompt, model_name, prices)
    except Exception as e:
        logger.error(f"[FAILED] {model_name}: {str(e)[:100]}")
        return None, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset Name (e.g., GSM8k)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel requests")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    
    # 1. 获取数据集路径
    if dataset_name not in DATASET_MAPPING:
        logger.error(f"Unknown dataset name: {dataset_name}. Available: {list(DATASET_MAPPING.keys())}")
        sys.exit(1)
    
    dataset_path = DATASET_MAPPING[dataset_name]
    
    # Initialize Template Manager
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_manager = TemplateManager(base_dir)

    # 2. 获取价格配置，如果没有配置则默认 0
    prices = MODEL_PRICING.get(model_name, {"in": 0, "out": 0})

    # 3. 准备文件路径
    # Create output directory: BaseModel_Output/{dataset_name}
    output_dir = os.path.join("BaseModel_Output", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    safe_model_name = model_name.replace("/", "_")
    output_file = os.path.join(output_dir, f"result_{safe_model_name}_{dataset_name}.jsonl")

    logger.info(f"STARTING TASK: Model=[{model_name}] Dataset=[{dataset_name}] Output=[{output_file}]")

    # 4. 读取数据
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # 5. 断点续传检查
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except: continue

    # 6. 执行循环
    def get_unique_id(item, idx):
        if dataset_name == "IFEVAL":
            return item.get("key")
        elif dataset_name == "Human_Eval":
            return item.get("task_id")
        return item.get("id", idx)

    def process_item(idx, item):
        # Use provided ID or generate one sequentially
        item_id = get_unique_id(item, idx)
        
        # Double check inside thread, though we filter before submit too
        if item_id in processed_ids:
            return None

        raw_question = item.get("content") or item.get("text") or item.get("question") or item.get("prompt")
        ground_truth = item.get("answer", "")
        
        # Apply template
        prompt = None
        if "GSM8k" in dataset_name:
            prompt = template_manager.get_prompt(dataset_name, raw_question)
        
        elif dataset_name == "MMLU_PRO":
            demonstration = item.get("prefix", "")
            prompt = template_manager.get_MMLUPro_prompt(raw_question, demonstration)

        elif dataset_name == "BBH":
            demonstration = item.get("prefix", "")
            prompt = template_manager.get_BBH_prompt(raw_question, demonstration)
        
        elif dataset_name == "IFEVAL":
            prompt = item.get("prompt")
        
        elif dataset_name == "Human_Eval":
            prompt = item.get("prompt")
        
        if prompt is None:
             return None

        response, cost = safe_run_single(prompt, model_name, prices)
        
        if response is not None:
            record = None
            if dataset_name == "MMLU_PRO" or dataset_name == "BBH":
                category = item.get("category")
                record = {
                    "id": item_id,
                    "model": model_name,
                    "dataset": dataset_name,
                    "category":category,
                    "original_prompt": raw_question,
                    "response": response,
                    "answer": ground_truth,
                    "cost": cost
                }
            elif "GSM8k" in dataset_name:
                record = {
                    "id": item_id,
                    "model": model_name,
                    "dataset": dataset_name,
                    "original_prompt": raw_question,
                    "response": response,
                    "answer": ground_truth,
                    "cost": cost
                }
            elif dataset_name == "IFEVAL":
                record = {
                    "id": item.get("key"),
                    "model": model_name,
                    "dataset": dataset_name,
                    "original_prompt": raw_question,
                    "response": response,
                    "cost": cost,
                    "instruction_id_list": item.get("instruction_id_list"),
                    "kwargs": item.get("kwargs"),
                }

            elif dataset_name == "Human_Eval":
                record = {
                    "id": item.get("task_id"),
                    "task_id": item.get("task_id"),
                    "model": model_name,
                    "dataset": dataset_name,
                    "original_prompt": raw_question,
                    "response": response,
                    "cost": cost,
                }
                
            return record
        return None

    with open(output_file, 'a', encoding='utf-8', buffering=1) as out_f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            total_items = len(data)
            batch_size = args.concurrency
            
            for i in range(0, total_items, batch_size):
                batch_futures = []
                # Determine the range for the current batch
                end_idx = min(i + batch_size, total_items)
                current_batch_indices = range(i, end_idx)
                
                # Submit tasks for the current batch
                for idx in current_batch_indices:
                    item = data[idx]
                    item_id = get_unique_id(item, idx)
                    
                    if item_id in processed_ids:
                        continue
                    
                    batch_futures.append(executor.submit(process_item, idx, item))
                
                # Process results for the current batch in order
                if batch_futures:
                    for future in batch_futures:
                        try:
                            record = future.result(timeout=100)
                            if record:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        except TimeoutError:
                            logger.error("Task timed out after 100s, skipping...")
                        except Exception as e:
                            logger.error(f"Error processing item: {e}")
                    
                    # Ensure data is written to disk after each batch
                    out_f.flush()
                    os.fsync(out_f.fileno())

    # 7. 任务完成日志 (Shell 脚本可以通过 grep 这个关键词来确认任务结束)
    logger.info(f"TASK FINISHED: Model=[{model_name}] Dataset=[{dataset_name}]")

if __name__ == "__main__":
    main()