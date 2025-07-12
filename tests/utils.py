import os
from typing import BinaryIO
import re
import regex  # 添加regex模块导入
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk, special_tokens=["<|endoftext|>"]):
    """
    处理一个文本块，先按特殊标记分割，然后对每个子块进行预分词
    
    Args:
        chunk: 要处理的文本块
        special_tokens: 特殊标记列表，默认为["<|endoftext|>"]
        
    Returns:
        预分词后的token列表
    """
    re_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 构建用于分割的正则表达式，转义特殊字符并用|连接
    split_pattern = "|".join(map(re.escape, special_tokens))
    
    # 按特殊标记分割文本
    sub_chunks = regex.split(split_pattern, chunk)
    
    # 对每个子块进行预分词并合并结果
    all_tokens = []
    for sub_chunk in sub_chunks:
        if sub_chunk:  # 忽略空子块
            tokens = regex.findall(re_pattern, sub_chunk)
            all_tokens.extend(tokens)
    
    return all_tokens

def pre_tokenize(file_path: str, num_process: int, split_special_token: str = "<|endoftext|>") -> dict[str, int]:
    ## Usage
    with open(file_path, "rb") as f:  # 使用传入的file_path参数
        boundaries = find_chunk_boundaries(
            f, num_process, split_special_token.encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.

        # parallelize implementation
        chunks = []
        token_counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        
        special_tokens = [split_special_token]  # 可以根据需要添加更多特殊标记
        before_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_chunk, chunk, special_tokens) for chunk in chunks}
            for future in tqdm(as_completed(futures),total=len(futures)):
                result = future.result()  # 添加括号调用方法
                token_counts.update(result)
        after_time = time.time()
        print(f"It takes {after_time - before_time} seconds on pre-tokenize with {num_process} processes")
        return token_counts
if __name__ == "__main__":
    pre_tokenize("/Users/lyx/Downloads/work/projects/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",num_process=32)