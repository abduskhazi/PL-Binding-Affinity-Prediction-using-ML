import sys
import random

print("Enter file name to do test train split : ", file=sys.stderr)
file_name = input()

with open(file_name) as data_f:
    data_list = [r.strip() for r in data_f.readlines()]

random.shuffle(data_list)

num_points = int(len(data_list) * 0.9)
print(num_points)

train_data = data_list[:num_points]
test_data = data_list[num_points:]

train_data = "\n".join(train_data)
test_data = "\n".join(test_data)

with open("train_" + file_name, "w") as train_f:
    train_f.write(train_data)

with open("test_" + file_name, "w") as test_f:
    test_f.write(test_data)
