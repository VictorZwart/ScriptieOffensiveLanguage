#Results

## LinearSVC
### linearSVC V1
First time running "batch_2_train_victor.csv" Explicitness without preprocessing:

weighted F1: 0.76

### LinearSVC V2 real dev set
accuracy: 0.77
f-measure: 
NOT: 0.88
IMPLICIT: 0.36
EXPLICIT: 0.61

### LinearSVC V3 real dev set with PP
accuracy: 0.76
f-measure: 
NOT: 0.87
IMPLICIT: 0.33
EXPLICIT: 0.59


## BiLSTM

###BiLSTM V1
First time running without preprocessing for explicitness:

accuracy: 0.9426

###BiLSTM V2 on the real dev set
acc: 0.7933

###BiLSTM V3 real dev set with PP
acc:  74.34%

### BiLSTM on test set:
Accuracies:
- 0.7578
- 0.7415
- 0.7370
- 0.7532
- 0.7360 \
average: 0.7451

                precision    recall     f1-score   support

            0       0.68        0.97    0.80        2072
            1       0.31        0.06    0.10        702
            2       0.00        0.00    0.00        334
            
        accuracy                        0.66        3108
        macro avg   0.33        0.34    0.30        3108
        weighted avg0.52        0.66    0.55        3108


                precision    recall  f1-score   support

           0        0.67        0.99    0.80        2072
           1        0.36        0.03    0.06        702
           2        0.29        0.01    0.01        334

        accuracy                        0.66      3108
        macro avg   0.44        0.34    0.29      3108
        weighted avg0.56        0.66    0.55      3108

                precision    recall  f1-score   support

           0        0.67      1.00      0.80      2072
           1        0.00      0.00      0.00       702
           2        0.29      0.01      0.01       334

        accuracy                        0.67      3108
        macro avg   0.32        0.33    0.27      3108
        weighted avg0.48        0.67    0.53      3108

                precision    recall  f1-score   support

           0        0.67      1.00      0.80      2072
           1        0.00      0.00      0.00       702
           2        0.00      0.00      0.00       334

        accuracy                        0.67      3108
        macro avg   0.22        0.33    0.27      3108
        weighted avg0.44        0.67    0.53      3108

              precision    recall  f1-score   support

           0       0.67      0.99       0.80      2072
           1       0.35      0.02       0.03       702
           2       0.38      0.01       0.02       334

        accuracy                        0.67      3108
        macro avg   0.47      0.34      0.28      3108
        weighted avg0.57      0.67      0.54      3108
## CNN 

### CNN V1
First time running without preprocessing for explicitness:
accuracy: 0.7841

###CNN V2 on the real dev set
acc: 0.7990

###CNN V3 real dev set with PP
acc: 73.30%

### CNN On test set:
accuracies:
- 0.7724
- 0.7719
- 0.8350
- 0.7572
- 0.8697

avg: 0.80124

                precision    recall  f1-score   support

           0       0.68      0.96       0.79      2072
           1       0.33      0.06       0.10       702
           2       0.14      0.03       0.05       334

        accuracy                        0.65      3108
        macro avg   0.39      0.35      0.31      3108
        weighted avg0.54      0.65      0.56      3108

              precision    recall  f1-score   support

           0       0.69      0.90      0.78      2072
           1       0.31      0.04      0.07       702
           2       0.20      0.18      0.19       334

        accuracy                        0.63      3108
        macro avg   0.40      0.37      0.35      3108
        weighted avg0.55      0.63      0.56      3108

              precision    recall  f1-score   support

           0       0.72      0.77      0.75      2072
           1       0.30      0.25      0.27       702
           2       0.20      0.21      0.20       334

        accuracy                        0.59      3108
        macro avg   0.41      0.41      0.41      3108
        weighted avg0.57      0.59      0.58      3108

        precision    recall  f1-score   support

           0       0.69      0.91      0.79      2072
           1       0.31      0.12      0.17       702
           2       0.21      0.08      0.12       334

        accuracy                        0.64      3108
        macro avg   0.40      0.37      0.36      3108
        weighted avg0.56      0.64      0.58      3108

              precision    recall  f1-score   support

           0       0.70      0.84      0.77      2072
           1       0.36      0.16      0.22       702
           2       0.19      0.18      0.19       334

        accuracy                        0.62      3108
        macro avg   0.42      0.39      0.39      3108
        weighted avg0.57      0.62      0.58      3108
 

