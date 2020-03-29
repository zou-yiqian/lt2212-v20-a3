# LT2212 V20 Assignment 3
## Part 1 - creating the feature table
#### arguments
There are four command line arguments for the file a3_features.py
- "inputdir".It is the root of the author directories. I assumed that enron_sample is in the directory a3.
- "outputfile". It is the name of the output file containing the table of instances. You could write any filename with .csv. For example, a3.csv.
- "dims". It is the output feature dimensions.
- "--test". It is the percentage of instances to label as test. The type is int, and default is 20.

Example run: python a3_features.py enron_sample test.csv 50 --test 20
#### clean data
- remove punctuation
- lowercase the text
#### dimensionality reduction
- SVD was used.
## Part 2 - design and train the basic model
#### arguments
There are three command line arguments for the file a3_features.py
- "featurefile".It is the file containing the table of instances and features. It is the file we created in part1.
- "--epoch". It is the epoch size. The type is int, and default is 100.
- "--batch". It is the batch size. The type is int, and default is 20.

Example run: python a3_model.py a3.csv --epoch 50 --batch 20
#### result: without any non-linearity and no hidden layer
              precision    recall  f1-score   support

         0.0       0.67      0.20      0.31        10
         1.0       0.53      0.90      0.67        10

    accuracy                           0.55        20
   macro avg       0.60      0.55      0.49        20
weighted avg       0.60      0.55      0.49        20
## Part 3 - augment the model
#### arguments
Two more arguments in part 3.
- "--hidden". It is the size of the hidden layer. The type is int, and default is 0.
- "--activation". It is the name of the non-linearity.

Example run: python a3_model.py a3.csv --hidden 20 --activation 'relu' --epoch 50 --batch 20
#### result: hidden layer = 20
- Sigmoid activation
