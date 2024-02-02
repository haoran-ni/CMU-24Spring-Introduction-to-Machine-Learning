import sys
import csv
import numpy as np


def readData(path: str):
    features = []
    labels = []
    with open(path, 'r') as file:
        content = csv.reader(file, delimiter='\t')
        header = next(content)
        for i in content:
            labels.append(float(i.pop(-1)))
            features.append([float(feat) for feat in i])
    return header[:-1], np.array(features), np.array(labels)


def entropyCalc(Y: np.ndarray):
    """
        Calculate the entropy H(Y)
    """
    values, counts = np.unique(Y, return_counts=True)
    entropy = 0
    total = sum(counts)
    for count in counts:
        p = count/total
        entropy += - p * np.log2(p)
    return entropy


def entropyCalc2(Y: np.ndarray, X: np.ndarray):
    """
        Calculate the entropy H(Y|X)
    """
    # Sanity check
    Y = Y.reshape(-1)
    X = X.reshape(-1)
    if len(X) != len(Y):
        raise RuntimeError("The sizes don't match")
    
    val_X, count_X = np.unique(X, return_counts=True)
    tot_X = sum(count_X)
    entropy = 0
    for val, count in zip(val_X, count_X):
        p = count/tot_X
        id = np.where(X == val)
        Y_temp = Y[id]
        entropy += p * entropyCalc(Y_temp)
    return entropy


def mutualInfo(Y: np.ndarray, X: np.ndarray):
    """
        Calculate the mutual information I(Y;X)
    """
    return entropyCalc(Y) - entropyCalc2(Y,X)


def maxMutualInfo(X: np.ndarray, Y: np.ndarray):
    """
        Obtain the attribute with the maximum mutual information
        Return the column with lower index in case of a tie
    """
    attr_info = {}
    
    # Special case where X has shape (n,)
    if len(X.shape) == 1:
        X.reshape(-1,1)

    for col in range(X.shape[1]):
        attr = X[:,col]
        I = mutualInfo(Y, attr)
        attr_info[col] = I
    
    I_max = max(attr_info.values())
    for key, val in attr_info.items():
        if val == I_max:
            return key, val


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


def writeOutput(pred, path):
    with open(f"{path}",'w') as file:
        for i in pred:
            file.write(f"{int(i)}\n")


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


class Node:
    graphTrain = ""
    pred = None
    def __init__(self, md, v=None, d=0):
        self.attribute = None
        self.attr_name = None
        self.left = None
        self.right = None
        self.depth = d
        self.max_depth = md
        self.vote = v
    
    def train(self, X: np.ndarray, Y: np.ndarray, header):
        if Node.graphTrain == "":
            Node.graphTrain += f"[{np.count_nonzero(Y == 0)} 0/{np.count_nonzero(Y == 1)} 1]\n"
        
        # If run out of attributes
        if X.shape[1] == 0:
            self.vote = majorityVoter(Y)
        
        # If reach maximum depth
        elif self.depth == self.max_depth:
            self.vote = majorityVoter(Y)
        else:
            attr, I = maxMutualInfo(X, Y)
            
            # If labels are pure
            if I == 0:
                self.vote = Y[0]
            
            # Split the node
            else:
                self.attribute = attr
                self.attr_name = header[self.attribute]
                x = X[:, self.attribute]
                
                # Left branch: attr = 0
                id_0 = np.where(x == 0)
                X_0 = X[id_0]
                Y_0 = Y[id_0]
                Node.graphTrain += self.graphInfo(0,Y_0)
                X_0 = np.delete(X_0, self.attribute, axis=1)
                header_0 = header.copy()
                header_0.remove(self.attr_name)
                self.left = Node(d=self.depth+1, md=self.max_depth)
                self.left.train(X_0, Y_0, header_0)
                
                # Right branch: attr = 1
                id_1 = np.where(x == 1)
                X_1 = X[id_1]
                Y_1 = Y[id_1]
                Node.graphTrain += self.graphInfo(1,Y_1)
                X_1 = np.delete(X_1, self.attribute, axis=1)
                header_1 = header.copy()
                header_1.remove(self.attr_name)
                self.right = Node(d=self.depth+1, md=self.max_depth)
                self.right.train(X_1, Y_1, header_1)
            
    def predict(self, X: np.ndarray, id=None):
        # Initialization
        if id is None:
            dim = len(X)
            id = np.arange(dim)
            Node.pred = np.zeros(dim)
        
        # If reach leaf node
        if self.vote != None:
            Node.pred[id] = self.vote
        else:
            x = X[:, self.attribute]
            
            # Left branch
            id_0 = np.where(x == 0)
            abs_id_0 = id[id_0]
            X_0 = X[id_0]
            X_0 = np.delete(X_0, self.attribute, axis=1)
            self.left.predict(X_0, id=abs_id_0)
            
            # Right branch
            id_1 = np.where(x == 1)
            abs_id_1 = id[id_1]
            X_1 = X[id_1]
            X_1 = np.delete(X_1, self.attribute, axis=1)
            self.right.predict(X_1, id=abs_id_1)
        return Node.pred

    def graphInfo(self, branch, Y: np.ndarray):
        num_0 = np.count_nonzero(Y == 0)
        num_1 = np.count_nonzero(Y == 1)
        return f"{(self.depth+1)*'| '}{self.attr_name} = {branch}: [{num_0} 0/{num_1} 1]\n"
    
    def printGraph(self, path: str):
        with open(f"{path}",'w') as file:
            file.write(Node.graphTrain)
        
            
class DecisionTree:
    """
        This is just a nominal class for intuitive purpose.
        The Node class does everything...
    """
    def __init__(self, max_depth, header):
        self.depth = max_depth
        self.root = None
        self.header = header
    
    def train(self, X: np.ndarray, Y: np.ndarray):
        self.root = Node(md=self.depth)
        self.root.train(X, Y, self.header)
    
    def predict(self, X: np.ndarray):
        return self.root.predict(X)

    def printGraph(self, path: str):
        self.root.printGraph(path)
        

    
if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    print_out = sys.argv[7]
    
    head, X_train, Y_train = readData(train_input)
    head, X_test, Y_test = readData(test_input)
    
    # Train decision tree
    tree = DecisionTree(max_depth, head)
    tree.train(X_train, Y_train)
    
    # Predict labels
    pred_train = tree.predict(X_train)
    pred_test = tree.predict(X_test)
    
    # Write outputs
    writeOutput(pred_train, train_out)
    writeOutput(pred_test, test_out)
    
    # Write error rates
    train_error = errorRate(Y_train, pred_train)
    test_error = errorRate(Y_test, pred_test)
    with open(f"{metrics_out}",'w') as file:
        file.write(f"error(train): {train_error}\n")
        file.write(f"error(test): {test_error}")
    
    # Print out the tree
    tree.printGraph(print_out)