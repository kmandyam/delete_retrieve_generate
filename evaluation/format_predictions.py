"""
Converts our format of predictions into format preferred by Li et al. (2018)

first argument is the paper output for formatting purposes
second argument is the predictions file
"""
import sys

# create a set of all the words in the original attribute vocab

predictions = []
with open(sys.argv[2]) as preds_file:
    for l in preds_file:
        predictions.append(l.strip())

with open(sys.argv[1]) as attr_file:
    i = 0
    for l in attr_file:
        split = l.strip().split("\t")
        print(split[0] + "\t" + predictions[i] + "\t" + split[2])
        i += 1
