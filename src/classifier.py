# K-Nearest Neighbour classifier
# Jack Kaloger 2017
# Project 1 for COMP30027
import csv
import sys

ATTR = ["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
K = int(sys.argv[1])

# converts input to appropriate value
def clean(data):
    if(data == 'M'):
        return 1
    if(data == 'F'):
        return 0
    if(data == 'I'):
        return 0.5
    return float(data)

# normalises an instance vector
def normalise(instance):
    mag = 0
    for x in instance:
        mag += x**2

    mag = mag ** 1/2
    for x in instance:
        x *= mag
    return instance


# opens file filename, returns the 2-tuple [instances, labels] of cleaned and labelled
def preprocess_data(filename):
    # attribute titles
    data = []
    labels = []
    # open file
    with open(filename,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load into array
        for row in reader:
            data.append(row)
    data = [list(map(clean, x)) for x in data]
    for instance in data:
        labels.append(get_label(instance))
    # we dont want to keep the rings data... that would be pointless
    data = [x[:-2] for x in data]
    data = [normalise(x) for x in data]
    return [data, labels]


# returns a score based on similarity of two instances, according to
# distance/similarity metric defined by method string
def compare_instance(instance1, instance2, method):
    if(method == "Euclidean Distance"):
        return dist_euclid(instance1, instance2)

# returns the euclidean distance of two instances
def dist_euclid(instance1, instance2):
    sum = 0
    for i in range(len(instance1)):
        sum += (instance1[i] - instance2[i])**2
    return sum**1/2
    return 0

# returns cosine similarity of two instances
def sim_cosine(instance1, instance2):
    a = 0
    b = 0
    for x in instance1:
        a += x**2
    for y in instance2:
        b += y**2
    c = [x * y for x,y in list(zip(instance1, instance2))]
    b = b**1/2
    a = a**1/2

    return sum(c)/a*b

# returns the class label
def get_label(instance):
    # classification from project spec as follows
    # young : rings <= 10
    # old : rings >= 11
    if(instance[ATTR.index("Rings")] <= 10):
        return "young"
    else:
        return "old"

# returns a (training, test) 2-tuple
def partition(data_set, cutoff):
    # find the index of our cutoff point for training data
    c = int(len(data_set[0]) * cutoff)
    # load training data up to cutoff
    training = [data_set[0][:c], data_set[1][:c]]
    # load test data from cutoff (+1)
    test = [data_set[0][c+1:], data_set[1][c+1:]]
    return [training, test]

# returns a list of (class, score) 2-tuples, for each of the k best neighbours,
# according to dist/sim metric defined by method string, for the given instance
# from the test data set based on all instances in training dataset
def get_neighbours(instance, training_data_set, k, method):
    neighbours = []
    for training_instance, label in zip(training_data_set[0], training_data_set[1]):
        neighbours.append([label, method(instance, training_instance)])
    return sorted(neighbours, key=lambda x: x[1])[:K]

# finds the majority class of a set of neighbours
def majority(neighbours):
    young = 0
    old = 0
    for c in neighbours:
        if(c[0] == "young"):
            young += 1
        if(c[1] == "old"):
            old += 1
    return "young" if young > old else "old"

# returns a predicted class label, according to the given neighbours from list (class, score) 2-tuples
# + voting method defined by method string
def predict_class(neighbours, method):
    return method(neighbours)

#accepts 2 tuples, predicted classes and real classes
def accuracy(predicted, real):
    T = 0
    N = 0
    for p, r in zip(predicted, real):
        print(p + "," + r)
        if(p == r):
            T += 1
        else:
            N += 1
    if(N == 0):
        return 1
    else:
        return T/(T + N)

# returns calculated value of evaluation metric
# divides dataset into training+test splits
def evaluate(data_set, metric):
    predicted = []
    partitioned_data = partition(data_set, 0.5)
    print("looping thru instances")
    for instance in partitioned_data[1][0]:
        print('.')
        neighbours = get_neighbours(instance, partitioned_data[0], 8, dist_euclid)
        predicted.append(predict_class(neighbours, majority))
    return metric(predicted, partitioned_data[1][1])

# MAIN
data = preprocess_data("../abalone.data")
print(evaluate(data, accuracy))

