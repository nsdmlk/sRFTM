import json
import random

with open('data/train_augmented.jsonl') as f:
    data = [json.loads(l) for l in f]

variables = ['a', 'b', 'c', 'x', 'y', 'z', 'n', 'm', 'p', 'q', 'k', 'i', 'j', 'u', 'v']
numbers = ['0', '1', '2', '3', '4', '5', '7', '9', '10', '100']

augmented = []
for item in data:
    for _ in range(2):
        new_input = item['input']
        new_output = item['output']

        # Стратегия 1: замена переменной
        old_var = random.choice(variables)
        new_var = random.choice(variables)
        if old_var in new_input and new_var != old_var:
            new_input = new_input.replace(old_var, new_var)
            new_output = new_output.replace(old_var, new_var)

        # Стратегия 2: замена числа (30% случаев)
        if random.random() < 0.3:
            old_num = random.choice(numbers)
            new_num = random.choice(numbers)
            if old_num in new_input and new_num != old_num:
                new_input = new_input.replace(old_num, new_num)
                new_output = new_output.replace(old_num, new_num)

        # Стратегия 3: перестановка (только для + и *)
        if random.random() < 0.2 and '+' in new_input:
            parts = new_input.split('+', 1)
            if len(parts) == 2:
                new_input = parts[1].strip() + ' + ' + parts[0].strip()
                new_output = parts[1].strip() + ' + ' + parts[0].strip()

        augmented.append({"input": new_input, "output": new_output})

# Объединение + дедупликация
all_data = data + augmented
seen = set()
unique = []
for item in all_data:
    if item['input'] not in seen:
        seen.add(item['input'])
        unique.append(item)

with open('data/train_augmented_v2.jsonl', 'w') as f:
    for item in unique:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Original: {len(data)}")
print(f"Augmented: {len(augmented)}")
print(f"Total unique: {len(unique)}")