import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_excel("Clean Data.xlsx")
data = data.iloc[:,:-1]
del data[data.columns[0]]
x_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1:]

#print(x_train)
#print(y_train)

regr = RandomForestRegressor(n_estimators=300, random_state=None)
regr.fit(x_train,y_train.values.ravel())
#prediction = regr.predict(x_train)
#print(prediction)
rand_for_reg = 'Rf_model.sav'
pickle.dump(regr, open(rand_for_reg, 'wb'))