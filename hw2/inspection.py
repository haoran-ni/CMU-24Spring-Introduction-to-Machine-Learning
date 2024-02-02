import sys
import csv
import numpy as np


def readData(path: str):
    features = []
    labels = []
    with open(path, 'r') as file:
        content = csv.reader(file, delimiter='\t')
        next(content)
        for i in content:
            labels.append(i.pop(-1))
            features.append(i)
    return np.array(features), np.array(labels)


def entropyCalc(Y: np.ndarray):
    values, counts = np.unique(Y, return_counts=True)
    entropy = 0
    total = sum(counts)
    for count in counts:
        p = count/total
        entropy += - p * np.log2(p)
    return entropy


def majorityVoter(Y: np.ndarray):
    # Find maximum occurance
    counter = {}
    values, counts = np.unique(Y, return_counts=True)
    for value, count in zip(values, counts):
        counter[value] = count
    max_num = max(counter.values())
    
    # Return the corresponding label(s)
    majority = []
    for key, value in counter.items():
        if value == max_num:
            majority.append(key)
    
    # Resolve tie
    if len(majority) != 1:
        majority = max(majority)
    else:
        majority = majority[0]

    return majority


def errorRate(target: np.ndarray, prediction: np.ndarray):
    # Sanity check
    target = target.reshape(-1)
    prediction = prediction.reshape(-1)
    if len(target) != len(prediction):
        raise RuntimeError("The sizes don't match")
    
    counter = 0
    size = len(target)
    for i in range(size):
        if target[i] != prediction[i]:
            counter += 1
    return counter/size


def main():
    # Parse arguments
    input = sys.argv[1]
    output = sys.argv[2]
    
    # Read data
    features, labels = readData(input)
    
    # Majority Voter
    prediction = np.full(labels.shape, majorityVoter(labels))
    
    # Output metrics
    entropy = entropyCalc(labels)
    error = errorRate(labels, prediction)
    with open(f"{output}",'w') as file:
        file.write(f"entropy: {entropy}\n")
        file.write(f"error: {error}")


if __name__ == "__main__":
    main()