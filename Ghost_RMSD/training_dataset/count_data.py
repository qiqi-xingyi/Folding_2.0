# --*-- conding:utf-8 --*--
# @time:11/3/25 23:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:count_data.py



def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":

    file_path = "train.jsonl"

    num_lines = count_lines(file_path)
    print(f"File: {file_path}")
    print(f"Total lines: {num_lines}")

