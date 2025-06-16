import random 

with open('datasets/kaist-rgbt/train-all-04.txt', 'r') as f:
    lines = f.readlines()
    
random.shuffle(lines)
split_ratio = 0.8
split_idx = int(len(lines) * split_ratio)

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open('datasets/kaist-rgbt/train.txt', 'w') as f:
    f.writelines(train_lines)

with open('datasets/kaist-rgbt/val.txt', 'w') as f:
    f.writelines(val_lines)