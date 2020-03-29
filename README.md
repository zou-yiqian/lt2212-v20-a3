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
Sigmoid:

              precision    recall  f1-score   support

         0.0       0.50      0.11      0.18         9
         1.0       0.56      0.91      0.69        11

    accuracy                           0.55        20
   macro avg       0.53      0.51      0.44        20
weighted avg       0.53      0.55      0.46        20

Tahn:

              precision    recall  f1-score   support

         0.0       0.56      0.82      0.67        11
         1.0       0.50      0.22      0.31         9

    accuracy                           0.55        20
   macro avg       0.53      0.52      0.49        20
weighted avg       0.53      0.55      0.51        20

Relu:

              precision    recall  f1-score   support

         0.0       0.33      0.25      0.29         8
         1.0       0.57      0.67      0.62        12

    accuracy                           0.50        20
   macro avg       0.45      0.46      0.45        20
weighted avg       0.48      0.50      0.48        20
#### result: hidden layer = 40
Sigmoid:

            precision    recall  f1-score   support

         0.0       0.80      0.33      0.47        12
         1.0       0.47      0.88      0.61         8

    accuracy                           0.55        20
   macro avg       0.63      0.60      0.54        20
weighted avg       0.67      0.55      0.53        20

Tahn:

              precision    recall  f1-score   support

         0.0       0.67      1.00      0.80        10
         1.0       1.00      0.50      0.67        10

    accuracy                           0.75        20
   macro avg       0.83      0.75      0.73        20
weighted avg       0.83      0.75      0.73        20
Relu:

              precision    recall  f1-score   support

         0.0       0.50      0.67      0.57         9
         1.0       0.62      0.45      0.53        11

    accuracy                           0.55        20
   macro avg       0.56      0.56      0.55        20
weighted avg       0.57      0.55      0.55        20
#### result: hidden layer = 60
Sigmoid:

              precision    recall  f1-score   support

         0.0       0.80      0.36      0.50        11
         1.0       0.53      0.89      0.67         9

    accuracy                           0.60        20
   macro avg       0.67      0.63      0.58        20
weighted avg       0.68      0.60      0.58        20

Tahn:

              precision    recall  f1-score   support

         0.0       0.36      0.67      0.47         6
         1.0       0.78      0.50      0.61        14

    accuracy                           0.55        20
   macro avg       0.57      0.58      0.54        20
weighted avg       0.65      0.55      0.57        20

Relu:

              precision    recall  f1-score   support

         0.0       0.60      0.33      0.43         9
         1.0       0.60      0.82      0.69        11

    accuracy                           0.60        20
   macro avg       0.60      0.58      0.56        20
weighted avg       0.60      0.60      0.57        20
#### discussion
With the increasing hidden layer, the model performs better.
## Part Bonus - plotting
bonus.py is used to plotting the precision-recall curve.
The hidden layer sizes are 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 and 100.
bonus.png is also uploaded.
