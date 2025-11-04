# --*-- conding:utf-8 --*--
# @time:11/3/25 23:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:count_data.py


import sys

def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_lines.py <file.jsonl>")
    else:
        path = 'train.jsonl'
        num_lines = count_lines(path)
        print(f"Total lines: {num_lines}")
