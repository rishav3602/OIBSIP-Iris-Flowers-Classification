# importing required libraies
import pandas as pd
import numpy as np



## Data Ingestions step
data=pd.read_csv('notebooks/data/Iris.csv')
data.head()


## making a copy of original dataset so that it do not harm our original dataset.
df = data.copy()
df.head()


## Lets drop the id column
df=df.drop(labels=['Id'],axis=1)
df.head()


#dropping the duplicated values
df = df.drop_duplicates()


## segregate numerical and categorical columns
numerical_columns=df.columns[df.dtypes!='object']
categorical_columns=df.columns[df.dtypes=='object']



## correlation
import seaborn as sns
sns.heatmap(df[numerical_columns].corr(),annot=True)


# Seperate features and target 
X = df.drop(labels=['Species'],axis=1)
Y = df[['Species']]


# Define which columns should be encode and which should be scaled
numerical_cols = X.select_dtypes(exclude='object').columns


from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


## Numerical Pipeline
num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())

    ]

)

preprocessor=ColumnTransformer([
('num_pipeline',num_pipeline,numerical_cols)])


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=30)


X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# Predict from the test dataset
predictions = svn.predict(X_test)




# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)




# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))





X_new = np.array([[  5.3, 2.5, 1.2, 1.9 ],[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))





# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)





# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)





model.predict(X_new)
