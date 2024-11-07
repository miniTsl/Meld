# import json

# path = "examples/program_1_v1.json"

# with open(path, "r") as f:
#     program = json.load(f)

# print(program["user_inputs"])

# for index, task_config in program["tasks"].items():
#     print(index)

# test = {
#     "0": "a",
#     "1": "b",
#     "2": "c",
#     "00": "d",
#     "01": "e",
#     "02": "f",
#     "a0": "g",
#     "a1": "h"
# }

# for index, value in test.items():
#     print(index, value)

from itertools import product
list1 = ["a", "b"]
list2 = [1, 2]

for combi in product(product(list1, list2), repeat=2):
    print(combi)