# Learning Goal: 

# To achieve correct insights from the data we need to focus on the Data Preprocessing stage. Our output quality is always depend on the input quality. So we need to have extra focus on data preprocessing because if we give the garbage value as training data to train the model then model output will be also garbage. Biggest challenge in the data preprocessing is to handle the missing values. This should be fill with the correct techniques. So we will see how to create the pipeline for handling missing values.

import numpy as np
import pandas as pd


df = pd.read_csv("train.csv")

pd.set_option("display.max_columns", None)


df.head()

df.isnull().sum()

df.describe()

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



x_train = df.drop(columns = "SalePrice", axis=1)

x_train.head()

y_train = df["SalePrice"]

y_train.head()

x_test = df.copy()

x_test.head()

Total_Null = x_train.isnull().sum()

Num_Col = x_train.select_dtypes(["int64", "float64"]).columns
Cat_Col = x_train.select_dtypes(["O"]).columns

Num_Col

Num_Col.shape

len(Num_Col)

Cat_Col


len(Cat_Col)

Num_Col_missing  = [col for col in Num_Col if Total_Null[col]>0]

Num_Col_missing

Cat_Col_missing = [col for col in Cat_Col if Total_Null[col]>0]

Cat_Col_missing

Num_Median_Imputer = Pipeline(steps=[("MedianImp", SimpleImputer(strategy="median"))])

Cat_Mode_Imputer = Pipeline(steps=[("ModeImp", SimpleImputer(strategy="most_frequent"))])

Impute_Processor = ColumnTransformer(transformers=[("MedianImputer", Num_Median_Imputer, Num_Col_missing ),
                                ("ModeImputer",Cat_Mode_Imputer, Cat_Col_missing )])

Impute_Processor.fit(x_train)

Impute_Processor.transformers_

Impute_Processor.transform

Impute_Processor.named_transformers_["MedianImputer"].named_steps["MedianImp"].statistics_

df[Num_Col_missing].describe()

df[Num_Col_missing].median()

Impute_Processor.named_transformers_["ModeImputer"].named_steps["ModeImp"].statistics_

df[Cat_Col_missing].mode()

Imputed_x_train = Impute_Processor.transform(x_train)

Final_Imputed_x_train = pd.DataFrame(Imputed_x_train, columns=Num_Col_missing+Cat_Col_missing)

Final_Imputed_x_train.head()

Final_Imputed_x_train.isnull().sum()

x_train.describe()

x_train.update(Final_Imputed_x_train)

x_train.describe()

x_train.shape

x_train.head()

x_test.head()

x_train.shape

from matplotlib import pyplot as plt
import seaborn as sns


sns.pairplot(x_train[Num_Col_missing])

sns.pairplot(x_test[Num_Col_missing])

# Result:

# To achieve the useful and correct insights from the data we have focused on the data preprocessing stage. As we knew Our output quality is always depend on the input quality. So we have spent extra attention on data preprocessing. We can apply different techniques like Mean, Median, Mode/Most Frequent, Constant values but  here we have applied the Median technique for the numerical features and on the other hand we have applied Mode / Most frequently technique for the categorical features. We have created the automated pipeline to fill the Na/ missing values. While handling the missing values we have to take care of the actual distribution of the data set. Note: This is only for learning and practice purpose. 


