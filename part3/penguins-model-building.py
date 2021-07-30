import pandas as pd


data=pd.read_csv('AusRentsFinal (1).csv')

df = data.copy()
target = 'RentAmount'
encode = ['StreetName','Suburb','State','Type','RentMonth',]

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]



df['RentAmount'] = df['RentAmount'].apply(target_encode)

# Separating X and y
X = df.drop('RentAmount', axis=1)
Y = df['RentAmount']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('RentAmount_clf.pkl', 'wb'))

