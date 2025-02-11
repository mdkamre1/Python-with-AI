from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np


my_data = np.genfromtxt('linreg_data.csv',delimiter=',')
print(my_data)


xp=my_data[:,0]
yp=my_data[:,1]
print(xp)


xp=xp.reshape(-1,1)
yp=yp.reshape(-1,1)
print('xp=',xp)


regr = linear_model.LinearRegression()
regr.fit(xp, yp)

print('slop b=',regr.coef_)
print('intercept a=',regr.intercept_)

xval=np.full((1,1),0.5)
yval=regr.predict(xval)
print(yval)


yhat=regr.predict(xp)
print('yhat=',yhat)

print('mean Absolute Error:',metrics.mean_absolute_error(yp,yhat))
print('mean squared error:',metrics.mean_squared_error(yp,yhat))
print('root mean squared error:',np.sqrt(metrics.mean_squared_error(yp,yhat)))
print('R2 value:',metrics.r2_score(yp,yhat))
