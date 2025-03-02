import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Load and process data
df = pd.read_csv('DS203-audio-labels - Sheet1 (4).csv')
print(df.head())
print(df.tail())
df = df.drop(index=116)
df.replace('TRUE', 1, inplace=True)
df.replace('FALSE', 0, inplace=True)
categories = ['National Anthem', "Marathi ‘Bhav Geet’", 'Marathi Lavni', 'Asha Bhosale', 'Kishor Kumar', 'Michael Jackson']

# Count occurrences for each category and store locations
count = {category: 0 for category in categories}
location = {category: [] for category in categories}
for i in range(116):  
    for j in range(6):
        if df[categories[j]].iloc[i] == 1:
            count[categories[j]] += 1
            location[categories[j]].append(i)

# Print count of each category in the sheet
print("Actual count of each category:")
for category, cnt in count.items():
    print(f"{category}: {cnt}")

# Function to create true output based on permutation
def create_true_output(i, ii, iii, iv, v, vi):
    category_values = {
        'National Anthem': i,
        "Marathi ‘Bhav Geet’": ii,
        'Marathi Lavni': iii,
        'Asha Bhosale': iv,
        'Kishor Kumar': v,
        'Michael Jackson': vi
    }
    true_output = [
        next((category_values[category] for category in categories if j in location[category]), 0)
        for j in range(116)
    ]
    return true_output, category_values

# Function to check accuracy of output
def check_accuracy(output):
    output = list(np.array(output) + 1)
    true_outputs = []
    category_values = []

    for perm in permutations(range(1, 7), 6):  
        i, ii, iii, iv, v, vi = perm
        output_ = create_true_output(i, ii, iii, iv, v, vi)
        true_output = output_[0]
        category_value = output_[1]
        true_outputs.append(true_output)
        category_values.append(category_value)

    ideal_output = 0
    ideal_location = 0

    for i in range(len(true_outputs)):
        match_count = sum(true_outputs[i][j] == output[j] for j in range(116))
        if match_count > ideal_output:
            ideal_output = match_count
            ideal_location = i

    print("Best match count:", ideal_output)
    print("Index of best-matching true_output:", ideal_location)

    true_output = true_outputs[ideal_location]
    category_value = category_values[ideal_location]
    extras = 0
    total_val = [0 for _ in range(6)]
    true_val = [0 for _ in range(6)]
    acc = [0 for _ in range(6)]
    for i in range(116):
        if true_output[i] == 0:
            extras += 1
        else:
            if true_output[i] == output[i]:
                total_val[true_output[i] - 1] += 1
                true_val[true_output[i] - 1] += 1
            else:
                total_val[true_output[i] - 1] += 1
    acc = [true_val[i] / total_val[i] if total_val[i] != 0 else 0 for i in range(6)]
    print(acc)
    print(total_val)
    print(true_val)
    print(category_value)
    print("Extras (uncategorized):", extras)

    return {"acc": acc, "total_val": total_val, "true_val": true_val, "category_value": category_value, "extras": extras}

# Function to format accuracy report with added details
def format_accuracy(accuracy_dic):
    acc = accuracy_dic["acc"]
    total_val = accuracy_dic["total_val"]
    true_val = accuracy_dic["true_val"]
    category_value = accuracy_dic["category_value"]
    extras = accuracy_dic["extras"]

    # Sort categories by their assigned number in category_value
    sorted_categories = sorted(category_value, key=category_value.get)

    # Header with aligned columns
    print(f"{'Assigned Number':<15} | {'Category':<20} | {'Actual Count':>12} | {'Matched Count':>14} | {'Accuracy (%)':>12}")
    print("-" * 100)

    # Row output with aligned columns
    for i, category in enumerate(sorted_categories):
        number_assigned = category_value[category]
        actual_count = count[category]
        matched_count = true_val[i]
        artist_accuracy = acc[i] * 100 if acc[i] != 0 else 0
        print(f"{number_assigned:<15} | {category:<20} | {actual_count:>12} | {matched_count:>14} | {artist_accuracy:>11.2f}%")

    # Extras line
    print("\nExtras (uncategorized):", extras)

# Sample usage
# output = [0] * 116  # Placeholder output for example
# accuracy_output = check_accuracy(output)
# format_accuracy(accuracy_output)
