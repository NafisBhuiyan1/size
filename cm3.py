import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv("cm3.csv")

print(dataframe)

features = dataframe[["variance","skewness","curtosis","entropy"]]
target = dataframe["class"]


X_train,X_test,y_train,y_test = train_test_split(features,target,random_state=0,test_size=0.25)


# X = np.reshape(X_train.ravel(),(len(X_train),1))
# Y = np.reshape(target_train.ravel(),(len(target_train),1))

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# X_test = np.reshape(X_train.ravel(),(len(X_train),1))
# Y_test = np.reshape(target_test.ravel(),(len(target_test),1))

targetpred = classifier.predict(X_test)

predicted = []

for i in targetpred:
    if i == 0:
        predicted.append("No")
    else:
        predicted.append("Yes")

actual = []

for i in y_test.ravel():
    if i == 0:
        actual.append("No")
    else:
        actual.append("Yes")
        
cm = confusion_matrix(actual,predicted,["Yes","No"])

print(cm)
