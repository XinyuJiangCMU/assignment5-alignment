import pandas as pd
import os
import glob

# --- 配置区域 ---
# 1. 这里填你刚才下载的 Parquet 文件的所在目录
# 根据你之前的日志，它在 ~/.cache/modelscope/... 下面
# 我们用 glob 自动找，省得你复制一大串文件名
CACHE_DIR = "/home/jxy/.cache/modelscope/hub/datasets/AI-ModelScope/MATH-lighteval/data"

# 2. 输出目录
OUTPUT_DIR = "data/MATH"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_parquet_to_jsonl(split_name, output_filename):
    # 自动寻找对应的 parquet 文件
    # split_name 通常是 'test' 或 'train'
    search_pattern = os.path.join(CACHE_DIR, f"{split_name}-*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️  警告：在 {CACHE_DIR} 没找到 {split_name} 的数据文件！")
        return

    print(f"📖 正在读取 {split_name} 数据 (Parquet 格式)...")
    # Pandas 原生支持读取 Parquet，就像读 Excel 一样简单
    df = pd.read_parquet(files[0])
    
    # --- 关键步骤：查看原始列名 ---
    print(f"   原始字段: {df.columns.tolist()}")
    
    # --- 关键步骤：对齐作业格式 ---
    # 作业 Prompt 需要 {question}，而数据集里叫 problem
    # 作业 Grader 需要 ground truth，数据集里叫 solution
    rename_map = {
        "problem": "question", 
        "solution": "answer"
    }
    # 仅重命名存在的列
    df = df.rename(columns=rename_map)
    
    # 确保只保留我们需要的列，避免文件太大
    keep_cols = ["question", "answer", "level", "type"]
    # 过滤掉不存在的列（以防万一）
    final_cols = [c for c in keep_cols if c in df.columns]
    df = df[final_cols]
    
    # --- 保存为 JSONL ---
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    # orient='records', lines=True 就是标准的 jsonl 格式
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"✅ 转换成功！已保存到: {output_path}")
    print(f"   前两条数据预览:\n{df.head(2).to_json(orient='records', lines=True, force_ascii=False)}")
    print("-" * 50)

# 执行转换
print("🚀 开始处理数据...")

# 1. 把 test 集转为 validation.jsonl (作业 Baseline 用) [cite: 147]
convert_parquet_to_jsonl("test", "validation.jsonl")

# 2. 把 train 集转为 train.jsonl (后续 SFT/RL 用) [cite: 452]
convert_parquet_to_jsonl("train", "train.jsonl")



# 🚀 开始处理数据...
# 📖 正在读取 test 数据 (Parquet 格式)...
#    原始字段: ['problem', 'level', 'solution', 'type']
# ✅ 转换成功！已保存到: data/MATH/validation.jsonl
#    前两条数据预览:
# {"question":"How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?","answer":"The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes.","level":"Level 3","type":"Algebra"}
# {"question":"What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?","answer":"One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{1}{100}=36$, and $130\\%$ of 20 is $ 130\\cdot 20\\cdot\\frac{1}{100}=26$.  The difference between 36 and 26 is $\\boxed{10}$.","level":"Level 1","type":"Algebra"}

# --------------------------------------------------
# 📖 正在读取 train 数据 (Parquet 格式)...
#    原始字段: ['problem', 'level', 'solution', 'type']
# ✅ 转换成功！已保存到: data/MATH/train.jsonl
#    前两条数据预览:
# {"question":"Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).","answer":"For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$.","level":"Level 5","type":"Algebra"}
# {"question":"A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?","answer":"Let $x$ be the number of band members in each row for the original formation, when two are left over.  Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members.  Based on the first equation, we must have $rx$ less than 98.  We can guess and check some values of $r$ and $x$ in the last equation.  If $r=18$, then $x=7$, and $rx=126$ which is too big.  If $r=16$, then $x=6$, and $rx=96$, which is less than 98.  Checking back in the second formation, we see that $(16-2)(6+1)=14\\cdot 7=98$ as it should.  This is the best we can do, so the largest number of members the band could have is $\\boxed{98}$.","level":"Level 5","type":"Algebra"}

# --------------------------------------------------