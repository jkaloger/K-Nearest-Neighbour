# K-Nearest Neighbor classifier
# Jack Kaloger 2017
# Project 1 for COMP30027
import csv

# converts input to appropriate value
def clean(data):
    if(data == 'M'):
        return 1
    if(data == 'F'):
        return 0
    if(data == 'I'):
        return 2
    return float(data)

# opens file filename,
# returns dataset (+class labels) comprised of instances in the file (1/line)
def preprocess_data(filename):
    # attribute titles
    data = []
    # open file
    with open(filename,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load into array
        for row in reader:
            data.append(row)
    return [list(map(clean, x)) for x in data]


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

# returns a list of (class, score) 2-tuples, for each of the k best neighbors,
# according to dist/sim metric defined by method string, for the given instance
# from the test data set based on all instances in training dataset
def get_neighbours(instance, training_data_set, k, method):
    neighbours = []
    for training_instance in training_data_set:
        neighbours.append([training_instance[8], method(instance, training_instance)])
    return sorted(neighbours, key=lambda x: x[1])[:k]

# returns a predicted class label, according to the given neighbors from list (class, score) 2-tuples
# + voting method defined by method string
def predict_class(neighbors, method):
	#a
        print("hi")

# returns calculated value of evaluation metric
# divides dataset into trainin+test splits
def evaluate(data_set, metric):
	#
        print("hi")


# MAIN

attributes = [["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]]
data = preprocess_data("../abalone.data")

print(get_neighbours(data[0], data, 10, sim_cosine))

