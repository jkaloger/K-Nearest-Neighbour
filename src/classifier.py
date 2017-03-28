''' K-Nearest Neighbour classifier
    Jack Kaloger 2017
    Project 1 for COMP30027
    This was a fun project! I implemented a lot of eval/dist metrics
    to play with.. you can run it with the
    evaluate function using the follwing metrics
    'accuracy'
    'macro precision'
    'macro recall'
    'micro precision'
    'micro recall'
    'young-precision'
    'young-recall'
    'old-precision'
    'old-recall'
    'F1-score'
    to change the distance metric, set DISTMETRIC to:
    'manhattan distance'
    'euclidean distance'
    'cosine similarity'
'''
import csv # required to load in dataset

###############################################################################
# Constants
###############################################################################
ATTR = ["Sex", "Length", "Diameter", "Height", "Whole Weight",
        "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
RINGINDEX = ATTR.index("Rings")

K = 7 # k value
HOLDOUT = 0.67 # percentage of instances as training instances
DISTMETRIC = "euclidean distance"
VOTING = "majority"
BETA = 1 # beta value for F-Score

################################################################################
# Functions for dataset processing
################################################################################

''' returns the class label based on abalone-2 spec
'''
def get_label(instance):
    # classification from project spec as follows
    # young : rings <= 10
    # old : rings >= 11
    if(instance[RINGINDEX] <= 10):
        return "young"
    elif(instance[RINGINDEX] >= 11):
        return "old"
    else:
        return ""

''' converts nominal attributes to appropriate values, and
    numerical data to float
'''
def clean(data):
    # Male and female are equidistant from Infant
    if(data == 'M'):
        return 1
    if(data == 'F'):
        return 1
    if(data == 'I'):
        return 0
    return float(data)

''' normalises a feature vector
'''
def normalise(instance):
    mag = 0
    # find the magnitude of the feature vector
    for x in instance:
        mag += x**2
    mag = mag ** 1/2
    # multiply each attribute by the magnitude
    for x in instance:
        x *= mag
    return instance

''' opens the file filename, returns the 2-tuple [instances, labels]
    normalised data_set
'''
def preprocess_data(filename):
    data = []
    labels = []
    # open file
    with open(filename,"r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load as a list of instances
        for row in reader:
            data.append(row)
    # apply the clean function to every attribute of every instance
    data = [list(map(clean, x)) for x in data]
    # label all instances based on abalone-2
    for instance in data:
        labels.append(get_label(instance))
    # remove the rings attribute for evaluation
    # (it would be unfair for our knn function to know the age, when that is our
    #  objective)
    data = [x[:-2] for x in data]
    # normalise all attribute vectors
    data = [normalise(x) for x in data]
    return [data, labels]


################################################################################
# Distance and Similarity Metrics
################################################################################

''' returns the length of a feature vector
'''
def vect_len(instance):
    sum = 0
    for attr in instance:
        sum += attr
    return sum/len(instance)

''' returns the dot product of two feature vectors
'''
def dot_product(instance1, instance2):
    sum = 0
    for attr1, attr2 in zip(instance1, instance2):
        sum += attr1 * attr2
    return sum

''' returns a score based on similarity of two instances, according to
    distance/similarity metric defined by method string
'''
def compare_instance(instance1, instance2, method):
    if(method == "euclidean distance"):
        return dist_euclid(instance1, instance2)
    elif(method == "cosine similarity"):
        return sim_cosine(instance1, instance2)
    elif(method == "manhattan distance"):
        return dist_manhattan(instance1, instance2)

''' returns the euclidean distance of two instances
'''
def dist_euclid(instance1, instance2):
    sum = 0
    for attr1, attr2 in zip(instance1, instance2):
        sum += (attr1 - attr2)**2
    return sum**1/2
    return 0

''' returns cosine similarity of two instances
'''
def sim_cosine(instance1, instance2):
    a = dot_product(instance1, instance2)
    b = vect_len(instance1)
    c = vect_len(instance2)
    # return a -ve value so when sorting in get_neighbours, the most
    # similar will be at the start of the list
    return -(a/b*c)

''' returns the manhattan distance between the two instances
'''
def dist_manhattan(instance1, instance2):
    sum = 0
    for attr1,attr2 in zip(instance1,instance2):
        sum += abs(attr1 - attr2)
    return sum

################################################################################
# K-NN
################################################################################

''' returns a list of (class, score) 2-tuples, for each of the k nearest
    neighbours, according to dist/sim metric defined by method string,
    for the given instance from the test data set based on all instances in
    training dataset
'''
def get_neighbours(instance, training_data_set, k, method):
    neighbours = [] # keep a list of all neighbours
    for training_instance, label in zip(training_data_set[0], training_data_set[1]):
        # find the distance between our instance and the next training instance
        dist = compare_instance(instance, training_instance, method)
        neighbours.append([label, dist])
    # sort the list of neighbours by ditance and return the first
    # K neighbours in that list
    return sorted(neighbours, key=lambda x: x[1])[:k]


################################################################################
# Voting Functions
################################################################################
''' finds the majority class of a list of neighbours
'''
def majority(neighbours):
    young = 0
    old = 0
    # count the number of young/old neighbours
    for neighbour in neighbours:
        if(neighbour[0] == "young"):
            young += 1
        if(neighbour[0] == "old"):
            old += 1
    # return the most common label
    return "young" if young > old else "old"

''' returns a predicted class label, according to the given neighbours
    from list (class, score) 2-tuples and voting method defined by method string
'''
def predict_class(neighbours, method):
    if(method == "majority"):
        return majority(neighbours)
    else:
        return 0

################################################################################
# Evaluation Metrics
################################################################################

''' calculates accuracy of predicted class labels using the two lists of classes
    predicted and real
'''
def accuracy(predicted, real):
    T = 0
    N = 0
    for p, r in zip(predicted, real):
        if(p == r): # when we predicted correctly (TP OR TN)
            T += 1
        else: # (FP OR FN)
            N += 1
    return T/(T + N) # equiv to (TP+TN)/(TP+TN+FP+FN)


''' creates a multi-class confusion matrix
'''
def create_confusion_matrix(predicted, real):
    M = [[0,0],[0,0]] # Predicted across, Real down
    for p, r in zip(predicted, real):
        if(p == "young"):
            if(p == r):
                M[0][0] += 1 # young prediction was correct
            else:
                M[1][0] += 1 # young prediction was incorrect
        elif(p == "old"):
            if(p == r):
                M[0][1] += 1 # old prediction was correct
            else:
                M[1][1] += 1 # old prediction was incorrect
    return M

''' calculates precision of specified classification from confusion matrix
'''
def precision(M, c):
    if(c == "young"):
        if(M[0][0] == 0):
            if(M[1][0] == 0):
                return 1
            else:
                return 0
        return M[0][0]/(M[0][0]+M[1][0]) # TP/TP+FP
    elif(c == "old"):
        if(M[1][1] == 0):
            if(M[0][1] == 0):
                return 1
            else:
                return 0
        return M[1][1]/(M[1][1] + M[0][1]) #TP/TP+FP
    else:
        return 0

''' calculates macro precision from confusion matrix
'''
def macro_precision(M):
    sum = precision(M, "young") + precision(M, "old")
    return sum/2

''' calculates micro precision from confusion matrix
    (This is the same as micro_recall)
'''
def micro_precision(M):
    total = 0
    sum = M[0][0] + M[1][1]
    for row in M:
        for col in row:
            total += col
    return sum / total

''' calculates recall of specified classification from confusion matrix
'''
def recall(M, c):
    if(c == "young"):
        if(M[0][0] == 0):
            if(M[0][1] == 0):
                return 1
            else:
                return 0
        return M[0][0]/(M[0][0] + M[0][1]) # TP/TP+FN
    elif(c == "old"):
        if(M[1][1] == 0):
            if(M[1][0] == 0):
                return 1
            else:
                return 0
        return M[1][1]/(M[1][1] + M[1][0]) # TP/TP+FN
    else:
        return 0

''' calculates macro recall from confusion matrix
'''
def macro_recall(M):
    sum = recall(M, "young") + recall(M, "old")
    return sum / 2

''' calculates the micro recall from confusion matrix
    (This is the same as micro_precision)
'''
def micro_recall(M):
    total = 0
    sum = M[1][1] + M[0][0]
    for row in M:
        for col in row:
            total += col
    return sum / total


''' calculates the f score for a given Beta value and precision, recall values
'''
def F_score(B, P, R):
    return (1 + B**2)*((P*R)/(R + (B**2)*P))

''' partitions a data_set into training and test sets for use in evaluation
    for 0 < cutoff < 1
'''
def partition(data_set, cutoff):
    # find the index of our cutoff point for training data
    c = int(len(data_set[0]) * cutoff)
    # load training data up to cutoff
    training = [data_set[0][:c], data_set[1][:c]]
    # load test data from cutoff (+1)
    test = [data_set[0][c+1:], data_set[1][c+1:]]
    return [training, test]

''' returns calculated value of evaluation metric
    divides dataset into training+test splits
'''
def evaluate(data_set, metric):
    predicted = []
    partitioned_data = partition(data_set, HOLDOUT)
    for instance in partitioned_data[1][0]:
        neighbours = get_neighbours(instance, partitioned_data[0], K, DISTMETRIC)
        predicted.append(predict_class(neighbours, VOTING))
    M = create_confusion_matrix(predicted, partitioned_data[1][1])
    if(metric == "accuracy"):
        return accuracy(predicted, partitioned_data[1][1])
    elif(metric == "macro recall"):
        return macro_recall(M)
    elif(metric == "micro recall"):
        return micro_recall(M)
    elif(metric == "macro precision"):
        return macro_precision(M)
    elif(metric == "micro precision"):
        return micro_precision(M)
    elif(metric == "young-precision"):
        return precision(M, "young")
    elif(metric == "old-precision"):
        return precision(M, "old")
    elif(metric == "young-recall"):
        return recall(M, "young")
    elif(metric == "old-recall"):
        return recall(M, "old")
    elif(metric == "F1-score"):
        return F_score(1, macro_precision(M), macro_recall(M))
    else:
        return 0

# MAIN
if __name__ == "__main__":
    print(evaluate(preprocess_data("../abalone.data"), "accuracy"))
