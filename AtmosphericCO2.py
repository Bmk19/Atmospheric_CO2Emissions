# =================  Atmospheric CO2 Emissions ===================

#This program is to show the growth of CO2 Emissions since 1960, when the Mauna Loa records began being published.
#The data has been retrieved and copied from the following website: https://climate.nasa.gov/vital-signs/carbon-dioxide/
# where the data can be downloaded and viewed.
#The objective of this is to show how polynomial regression can look in relation to linear regression.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
x_line = [[1960], [1965], [1970], [1975], [1980], [1985], [1990], [1995], [2000], [2005], [2010], [2015],
          [2020]]  # Year.
y_line = [[316.19], [319.42], [325.13], [330.59], [338.32], [345.88], [354.33], [360.68], [369.67], [380.11], [389.79],
          [401.85], [412.43]]  # CO2 PPM

# Testing set
x_poly = [[1960], [1965], [1970], [1975], [1980], [1985], [1990], [1995], [2000], [2005], [2010], [2015],
          [2020]]  # Year
y_poly = [[316.19], [319.42], [325.13], [330.59], [338.32], [345.88], [354.33], [360.68], [369.67], [380.11], [389.79],
          [401.85], [412.43]]  # CO2 PPM

# Train the Linear Regression model and plot a prediction
regressor = LinearRegression()
regressor.fit(x_line, y_line)
xx = np.linspace(1950, 2020, 10)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Set the degree of the Polynomial Regression model
quadratic_degree = PolynomialFeatures(degree=2)

# This preprocessor transforms an input data matrix into a new data matrix of a given degree
x_train_quadratic = quadratic_degree.fit_transform(x_line)
x_test_quadratic = quadratic_degree.transform(x_poly)

# Train and test the regressor_quadratic model
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_line)
xx_quadratic = quadratic_degree.transform(xx.reshape(xx.shape[0], 1))

# Plot the graph
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle='--')
plt.title('Atmospheric Carbon Dioxide Emissions Over 60 Years')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions in Parts Per Million(PPM)')
plt.axis([1950, 2020, 310, 420])
plt.grid(True)
plt.scatter(x_line, y_line)
plt.show()
print(x_line)
print(x_train_quadratic)
print(x_poly)
print(x_test_quadratic)

# If you execute the code, you will see that the simple linear regression model is plotted with
# a solid line. The quadratic regression model is plotted with a dashed line and evidently
# the quadratic regression model fits the training data slightly better.
