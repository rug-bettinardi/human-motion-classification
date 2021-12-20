## Detection of Human Activity using Wrist-Accelerometer

Chech out the blog post --> [here](https://ruggerobettinardi.eu/blog/accelerometer-classification/)

### Aim:
Develop algorithm to classify accelerometer time series according to the corresponding type of motion / activity

### Dataset Description:
The Dataset for ADL (Activities of Daily Living) Recognition with Wrist-worn Accelerometer is a public collection of labelled accelerometer data recordings to be used for the creation and validation of supervised acceleration models of simple ADL.

It can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer).

The Dataset is composed of the recordings of 14 simple ADL (*brush_teeth, climb_stairs, comb_hair, descend_stairs, drink_glass, eat_meat, eat_soup, getup_bed, liedown_bed, pour_water, sitdown_chair, standup_chair, use_telephone, walk*) perfomed by a total of 16 volunteers. 
**Note** that 5 of those activities had less than 5 records each (*brush_teeth, eat_meat, eat_soup, use_telephone*), therefore I removed them from the analysis as they are too few to allow proper features learning. 

The data are collected by a single tri-axial accelerometer (x, y, z) attached to the right-wrist of the volunteer, sampled at 32 Hz.

--- 

### Approach

I will try to classify accelerometer's data into their corresponding activity in two ways: 
1. extracting a number of features I chose to compute to characterize each record's time series, and 
2. leveraging the amazing [tsfresh](https://tsfresh.readthedocs.io/en/latest/) library, which automatically extracts and [selects](https://tsfresh.readthedocs.io/en/latest/text/feature_filtering.html) from more than 2000 features specifically designed to summarize time series data.  

In both cases, I will (A) first extract a number of features characterizing each recorded time series, (B) afterward I will feed all those features to a set of classifiers to see which perform best, and finally (C) I will build an ensemble classifier by stacking those classifiers having best performances.
Model Performance

---

### Model Performance

I will assess the performance of each classifier using the [macro F1-score](https://en.wikipedia.org/wiki/F-score), a metric which summarizes both the precision and the recall (see explanation below): in fact the F1-score is the *harmonic mean* between the two, averaged over all classes to be predicted.

A perfect model has an F1-score of 1.

- **Precision**:  the fraction of true positive examples among the examples that the model classified as positive. In other words, the number of true positives divided by the number of false positives plus true positives. 


- **Recall**: also known as sensitivity, is the fraction of examples classified as positive, among the total number of positive examples. In other words, the number of true positives divided by the number of true positives plus false negatives.


---

### References

- Bruno, B., Mastrogiovanni, F., Sgorbissa, A., Vernazza, T., Zaccaria, R.:
Analysis of human behavior recognition algorithms based on acceleration data.
In: IEEE Int Conf on Robotics and Automation (ICRA),
pp. 1602--1607 (2013)

- Bruno, B., Mastrogiovanni, F., Sgorbissa, A., Vernazza, T., Zaccaria, R.:
Human motion modelling and recognition: A computational approach.
In: IEEE Int Conf on Automation Science and Engineering (CASE),
pp. 156--161 (2012)
