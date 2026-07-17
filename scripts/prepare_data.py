import os
import numpy as np
import json
from tqdm import tqdm

from cs336_basics.tokenizer import train_bpe, Tokenizer

def main():
    text_file = "../data/TinyStoriesV2-GPT4-valid.txt"
    output_npy = "../data/tinystories_valid_tokens.npy"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print(f"1. 开始在 {text_file} 上训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    vocab, merges = train_bpe(
        input_path=text_file, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens
    )
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    
    # with open("vocab.json", "w") as f:
    #     json.dump({k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f)
    
    print("2. 分词器训练完成！开始读取全文进行预处理分词...")
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("3. 正在编码全文为 Token IDs (这需要一些时间)...")
    tokens = tokenizer.encode(text)
    
    print(f"4. 编码完成！共获得 {len(tokens):,} 个 Tokens。正在保存为 NumPy 数组...")
    tokens_np = np.array(tokens, dtype=np.uint16) 
    
    np.save(output_npy, tokens_np)
    print(f"✅ 处理完毕！数据已保存至 {output_npy}")
    print(f"文件大小约为: {os.path.getsize(output_npy) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()