

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
np.random.seed(2809)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

X_train = train_data.iloc[:, 3:].values
y_train = train_data.iloc[:, 1].values
X_test = test_data.iloc[:, 2:].values


model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, C = 10, l1_ratio = 0.1)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]


## Save predictions to file
output = pd.DataFrame(data={'id': test_data["id"], 'prob': y_pred})
output.to_csv("mysubmission.txt",index=False, sep=',', header=True)
