# K-Nearest Neighbor classifier
# Jack Kaloger 2017
# Project 1 for COMP30027
import csv

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
    return(data)


# returns a score based on similarity of two instances, according to
# distance/similarity metric defined by method string
def compare_instance(instance1, instance2, method):
        print("hi")

# returns a list of (class, score) 2-tuples, for each of the k best neighbors,
# according to dist/sim metric defined by method string, for the given instance
# from the test data set based on all instances in training dataset
def get_neighbours(instance, training_data_set, k, method):
	#
        print("hi")

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
for row in data:
    print(row)
