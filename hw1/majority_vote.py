import sys
import csv


class MajorityVoter:
    def __init__(self):
        self.majority = None
    
    def train(self, train_features, train_labels):
        # Count labels
        counter = {}
        for label in set(train_labels):
            num = train_labels.count(label)
            counter[label] = num
        max_num = max(counter.values())
        
        # Return the majority label
        maj_label = []
        for y, n in counter.items():
            if n == max_num:
                maj_label.append(y)
        if len(maj_label) != 1:
            # Pick the numerically higher value if tie
            self.majority = max(maj_label)
        else:
            self.majority = maj_label[0]
    
    def predict(self, features):
        # Always predict the majority value
        prediction = []
        for i in features:
            prediction.append(self.majority)
        return prediction


def read_data(path: str):
    features = []
    labels = []
    with open(path, 'r') as file:
        content = csv.reader(file, delimiter='\t')
        next(content)
        for i in content:
            labels.append(i.pop(-1))
            features.append(i)
    return features, labels


def metrics(target, prediction):
    if len(target) != len(prediction):
        raise RuntimeError("The sizes don't match")
    counter = 0
    size = len(target)
    for i in range(size):
        if target[i] != prediction[i]:
            counter += 1
    return counter/size


def write_pred_file(path: str, prediction):
    with open(f"{path}",'w') as file:
        for i in prediction:
            file.write(f"{i}\n")


if __name__ == '__main__':
    # Parse arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    
    # Read data
    train_features, train_labels = read_data(train_input)
    test_features, test_labels = read_data(test_input)
    
    # Training majority voter
    model = MajorityVoter()
    model.train(train_features, train_labels)
    
    # Predictions
    train_pred = model.predict(train_features)
    test_pred = model.predict(test_features)
    
    # Output prediction files
    write_pred_file(train_out, train_pred)
    write_pred_file(test_out, test_pred)

    # Output metrics
    train_metrics = metrics(train_labels, train_pred)
    test_metrics = metrics(test_labels, test_pred)
    with open(f"{metrics_out}",'w') as file:
        file.write(f"error(train): {train_metrics}\n")
        file.write(f"error(test): {test_metrics}")