# K-Nearest Neighbor classifier
# Jack Kaloger 2017
# Project 1 for COMP30027

# opens file filename,
# returns dataset (+class labels) comprised of instances in the file (1/line)
def preprocess_data(filename):
    #

# returns a score based on similarity of two instances, according to
# distance/similarity metric defined by method string
def compare_instance(instance, instance, method):
    #

# returns a list of (class, score) 2-tuples, for each of the k best neighbors,
# according to dist/sim metric defined by method string, for the given instance
# from the test data set based on all instances in training dataset
def get_neighbours(instance, training_data_set, k, method):
    #

# returns a predicted class label, according to the given neighbors from list (class, score) 2-tuples
# + voting method defined by method string
def predict_class(neighbors, method):
    #

# returns calculated value of evaluation metric
# divides dataset into trainin+test splits
def evaluate(data_set, metric):
    #
