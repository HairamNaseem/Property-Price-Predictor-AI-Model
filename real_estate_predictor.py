import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
1.  Id	         To count the records.
2	MSSubClass	 Identifies the type of dwelling involved in the sale.
3	MSZoning	 Identifies the general zoning classification of the sale.
4	LotArea	     Lot size in square feet.
5	LotConfig	 Configuration of the lot
6	BldgType	 Type of dwelling
7	OverallCond	 Rates the overall condition of the house
8	YearBuilt	 Original construction year
9	YearRemodAdd Remodel date (same as construction date if no remodeling or additions).
10	Exterior1st	 Exterior covering on house
11	BsmtFinSF2	 Type 2 finished square feet.
12	TotalBsmtSF	 Total square feet of basement area
13	SalePrice	 To be predicted
'''

houses = pd.read_csv(r'C:\Users\haira\OneDrive\Desktop\Fast University\Semester-3\Programming for AI\HousePricePrediction.xlsx - Sheet1.csv')
print(houses.head(5))

#Histogram of data
houses.hist(bins=50,figsize=(5,5))
plt.show()

# DATA CLEANING
print(houses.info)
print(houses.isnull().sum())
houses['SalePrice'] = houses['SalePrice'].fillna(houses['SalePrice'].mean())
houses = houses.dropna()
print(houses.isnull().sum())
from sklearn.preprocessing import OneHotEncoder
type = (houses.dtypes == 'object')
object_cols = list(type[type].index)
print("\nCategorical variables:")
print(object_cols)
print('No. of. categorical features: ', len(object_cols))
one_hot = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(one_hot.fit_transform(houses[object_cols]))
OH_cols.index = houses.index
OH_cols.columns = one_hot.get_feature_names_out()
houses = houses.drop(object_cols, axis=1)
houses = pd.concat([houses, OH_cols], axis=1)


# TRAIN-TEST SPLITTING
def split_train_test(houses, test_ratio):
    np.random.seed(42)  # random values will not change every time code is run
    shuffled = np.random.permutation(len(houses))
    test_set_size = int(len(houses) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return houses.iloc[train_indices], houses.iloc[test_indices]
train_set, test_set = split_train_test(houses, 0.2)  # 20% for test and 80% for train
print(f"\nrows in train set: {len(train_set)}\nrows in test set: {len(test_set)}")
houses_tr = train_set.copy()

# CORRELATIONS IN DATA
corr_matrix = houses.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)
print('\n\nCORRELATION MATRIX\n')
print(corr_matrix)

plt.scatter(houses['YearBuilt'], houses['SalePrice'], color='red',linestyle='--',s=5)
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.show()

plt.scatter(houses['LotArea'], houses['SalePrice'], color='blue',linestyle='--',s=5)
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.show()

# TRYING ATTRIBUTE COMBINATIONS
houses['PRSQ'] = houses['SalePrice'] / houses['LotArea']  # calculating price per square feet
print("\n\nTRAIN SET AFTER ADDING NEW COLUMN OF PRICE PER SQUARE FEET\n")
print(houses_tr.describe())
houses = train_set.drop('SalePrice', axis=1)

houses_labels = train_set['SalePrice'].copy()

# FEATURE ENGINEERING - CREATING A PIPELINE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scalar', StandardScaler()),
])
houses_trained = my_pipeline.fit_transform(houses)
print("\n\nCONVERTING DATA INTO ARRAY THROUGH PIPELINING\n")
print(houses_trained.shape)
print(houses_trained)

# SELECTING A DESIRED MODEL - Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(houses_trained, houses_labels)
data = houses.iloc[:90]
labels = houses_labels.iloc[:90]
prepared_data = my_pipeline.transform(data)
print("\n\n SAMPLE PREDICTED VALUES\n")
print("predicted values: ", model.predict(prepared_data))
print("actual values:\n", labels)



# EVALUATING THE MODEL
from sklearn.metrics import mean_squared_error
houses_predictions = model.predict(houses_trained)
mse = mean_squared_error(houses_labels, houses_predictions)
rmse = np.sqrt(mse)
print("\nRoot mean squared error: ", rmse)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, houses_trained, houses_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print("\nscores: ", rmse_scores)


def print_scores(scores):
    print("\nmean: ", scores.mean())
    print("standard deviation: ", scores.std())


print_scores(rmse_scores)

X_test = test_set.drop('SalePrice', axis=1)
Y_test = test_set['SalePrice'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("\nFINAL PREDICTIONS: ", final_predictions, Y_test)
print("\nFINAL ERROR MARGIN: ", final_rmse)
print("\nAVERAGE PREDICTION IS: ",np.mean(final_predictions)," ± ",final_rmse)

