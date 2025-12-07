import os
import json
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- 1. 配置路径 (请修改为你自己的真实路径) ---
# 模型绝对路径 (刚才用 python 命令查到的那个)
MODEL_PATH = "/home/jxy/.cache/modelscope/hub/Qwen/Qwen2.5-Math-1.5B" 

# 数据集路径
DATA_PATH = "data/MATH/validation.jsonl"

# 结果保存路径
OUTPUT_FILE = "baseline_results.jsonl"

# --- 2. 导入作业自带的评分工具 ---
# 确保你在 assignment5-alignment 根目录下运行，否则找不到这个包
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("❌ 错误：找不到 cs336_alignment 模块。请确保你在作业根目录下运行此脚本。")
    exit(1)

def load_data(filepath):
    prompts = []
    ground_truths = []
    print(f"正在读取数据: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 作业提供的 R1-Zero Prompt 模版文件
            # 我们直接手动构建，防止文件读取路径错误
            # 格式参考: cs336_alignment/prompts/r1_zero.prompt
            prompt_content = (
                "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
                "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
                "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
                "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
                f"User: {item['question']}\n"
                "Assistant: <think>"
            )
            prompts.append(prompt_content)
            ground_truths.append(item['answer'])
    return prompts, ground_truths

def main():
    # 1. 准备数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到数据文件 {DATA_PATH}。请检查第一步是否下载成功。")
        return
    
    prompts, ground_truths = load_data(DATA_PATH)
    print(f"加载了 {len(prompts)} 条测试数据。")

    # 2. 初始化 vLLM
    print(f"正在加载模型: {MODEL_PATH} ...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1, # 单卡运行。如果你想用2张卡推理，改成2
        gpu_memory_utilization=0.90, # 显存占用率
        dtype="bfloat16" # 3090 必须用 bf16
    )

    # 3. 设置采样参数 (作业要求)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"], # 关键：遇到答案结束标签停止
        include_stop_str_in_output=True # 保留 </answer> 以便解析
    )

    # 4. 执行推理 (Batch Inference)
    print("🚀 开始推理 (Generating)...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 评分与统计
    print("正在评分 (Grading)...")
    results = []
    correct_count = 0
    format_error_count = 0

    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]
        
        # 拼接完整的生成内容 (Prompt最后的 <think> + 生成的内容)
        # 注意：vLLM 生成的是 Assistant 后面的内容，我们需要把标签补全方便解析
        # 其实 r1_zero_reward_fn 只需要生成部分的 string
        # 但为了保险，我们把生成的直接传进去，作业的 grader 应该能处理
        
        # 调用作业自带的评分函数
        # 输入：模型生成的文本，标准答案
        # 输出：{'reward': 1.0/0.0, 'format_reward': 1/0, 'answer_reward': 1/0}
        
        # 这里有个小坑：作业提示词末尾是 "Assistant: <think>"
        # 模型生成的是 "xxxx </think> <answer> yyyy </answer>"
        # 为了让解析器工作，我们最好把开头的 "<think>" 补回去传给 grader
        full_response = "<think>" + generated_text
        
        score = r1_zero_reward_fn(full_response, ground_truth)
        
        if score['reward'] == 1.0:
            correct_count += 1
        if score['format_reward'] == 0.0:
            format_error_count += 1
            
        results.append({
            "question": prompts[i],
            "ground_truth": ground_truth,
            "generated": full_response,
            "score": score
        })

    # 6. 保存结果与输出报告
    accuracy = correct_count / len(prompts)
    format_error_rate = format_error_count / len(prompts)
    
    print("-" * 30)
    print(f"✅ 评估完成！")
    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"格式错误率 (Format Error): {format_error_rate:.2%}")
    print(f"详细结果已保存至: {OUTPUT_FILE}")
    print("-" * 30)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()