import pandas as pd
import numpy as np
url='https://raw.githubusercontent.com/pooja28567/mlcode1/main/project1.csv'
food_data = pd.read_csv(url, error_bad_lines=False)
from sklearn.model_selection import train_test_split
x=food_data.drop('Target',axis=1)
y=food_data['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
print('Ammonia ={} ppm ,Oxygen={} ppm ,Ethelyene={} ppm'.format(32,3,323))
a = [[32,3,323]]
target1 = logmodel.predict(a)
if target1== 0:
    print('Food Sample is not spoilt')
else:
    print('Food Sample is spoilt')