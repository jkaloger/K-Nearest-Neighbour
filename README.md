# k-nn
simple k-nearest neighbour classifier.
tested using the [abalone](https://archive.ics.uci.edu/ml/datasets/Abalone) dataset from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/index.html)

# report
## Introduction
This report outlines the key insights gained from programming my K-Nearest Neighbour algorithm. In it, I run several tests on similarity and distance metrics, evaluation functions, validation frameworks, dataset representations and k values to find the “best” combination of the tools learned so far in the subject. I used the 2 class representation of the dataset, I believe these thresholds were chosen based on the number of Infant instances vs M/F instances for those rings. 
## Section 1 – Similarity / Distance Metrics
I implemented 3 Similarity/Distance metrics: Euclidean, Manhattan and Cosine. I chose these because they deal with continuous values, not sets and therefore didn’t require changing the representation of the dataset, which could potentially lead to problems including overfitting.
## Section 2 – Validation
I used a Holdout strategy for validation. During program execution, the dataset is split into two sections; training and testing datasets. While it’s a simpler implementation than Cross Validation it has many caveats – the most prevalent is overfitting. If the sample used for training is not random enough or not large enough it may not generalise well. Additionally, holdout has the problem of test and training data competing for the finite amount of data available – we may get greater precision but less accuracy, for example, with a smaller test size.
## Section 3.1 – Representation
Choosing the right representation for the data set is important – potentially improving the effectiveness of the implementation. Different representations can enable the use of different similarity metrics such as Jaccard, or greater weighting on “better” attributes.
Gender is the only nominal attribute. Intuitively, I assumed Male and Female as equidistant from Infant; I chose the values M=1, I=0, F=1 as abalone that are infants are young, and those that have gender, usually old. The distance is arbitrary, because each feature vector is normalised during execution. 
Normalisation allows the system to give measurements with differing units equal impact on the overall vector. For example, in our dataset ‘whole weight’ attributes can range from (0.002, 2.8255), whereas length has range (0.075, 0.815). Since we want these attributes to have equal weighting, we normalise the whole vector. Of course, feature weighting could be introduced, but only after normalisation.
Measuring the “goodness” of a feature in this dataset is difficult, not only in that this measurement is vague but additionally as the dataset is small and we’re using the Holdout strategy (which can lead to overfitting). Moreover, since the attributes in the set are mostly continuous, calculations involving pointwise mutual information are difficult to setup in a meaningful way. Finding features that are correlated with classes is more relevant. 
I setup a test that shows the precision of a dataset after removing a certain feature. If precision decreases – that is, the positive predictive value of the algorithm decreases – the feature must be well correlated with the class. Figure 1 shows just this, using Manhattan distance and K=21
![figure 1](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure1.png)
Figure 1 shows the difference between regular representation and representation after removing that attribute. Clearly, each feature has as much weighting as another – the correlation between removing attributes and precision of old labels is likely due to the overall lower precision of old labelling by the algorithm. 
Moreover, precision for young labels is lowered for all attributes, indicating removing any of them would not be beneficial to the precision of the algorithm. In other words, all features in the vector contribute to our prediction of the class in equal measure. This is intuitive, looking at the features – the physical dimensions of any animal are likely to help guess its age.

## 3.2 – Similarity Metric
Gauging the most effective similarity/distance metrics involved running the program several times with different metrics. For standardisation, I ran the algorithm each time with k=21 (arbitrary), using the holdout strategy to generate accuracy, macro precision, macro recall and F1-Score metrics.
![figure 2](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure2.png)
Figure 2 does not show a clear winner. While Euclidean distance and Manhattan distance are more accurate – 0.7521, 0.7464 vs 0.6570 – cosine similarity is far more precise – 0.8285 vs 0.5548, 0.5634. This data suggests that for precision choose Cosine Similarity and for Accuracy choose Euclidean distance. Interestingly, the F1 Score of cosine similarity is greater than Euclidean and Manhattan distances’. Looking closely, this is because of cosine similarity’s far greater precision. Below, figure 3 demonstrates that cosine similarity has a precision of 1 for old labels. 
![figure 3](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure3.png)
I discovered that this is because cosine similarity never predicted young labels – and so was 100% precise; however, it began classifying young classes at very high k values. This anomaly was due to overfitting in the training set. I conclude that Euclidean distance is the “best” metric for this dataset. It has higher accuracy and recall than cosine similarity at lower k values (which we use, seen in the next section.)
## 3.3 – K value
Evaluating the “best” performing k value, I ran the algorithm using holdout (67% training, 33% test) and the Euclidean distance metric for (some arbitrary set) k ∈ {0, 1, 2, 3, …, 100, 101} to evaluate Accuracy and F-Score with β = 1.
![figure 4](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure4.png)
Figure 4 shows an increase in accuracy until k=11, reaching ~0.76 and fluctuating by ±0.01 from there. Clearly, any k Value above 11 is not useful in terms of accuracy. Increasing k values above ~2 correlates with a decrease in F1 Score. This is most likely due to low Precision of the Euclidean distance metric, demonstrated in figure 5. Increasing k →11 reduces the error rate.
![figure 5](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure5.png)
To verify my findings, I also tested arbitrarily large values k ∈{1250, 1251, … 1275}. The result, Figure 6, shows large values continuing to decrease away from ~0.76. Consequently, it is safe to conclude that the “best” k value to use with this dataset is k≈11.
![figure 6](https://github.com/jkaloger/K-Nearest-Neighbour/raw/master/docs/figure6.png)
## Conclusion
With the evidence gathered through these tests, the optimum k-nn algorithm implementation for this dataset appears to be k=11 using Euclidean Distance. The main concern is that the experimentation was done using a holdout strategy, which as discussed above could (most likely) lead to overfitting.
