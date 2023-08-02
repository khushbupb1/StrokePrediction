#!/usr/bin/env python
# coding: utf-8

# # Import all imp Libraries

# In[1]:


# import important libraries to use diffrent functions 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# This dataset is used to predict whether a patient is likely to get stroke based on the input parameters 
# like gender, age, various diseases, and smoking status. 
# Each row in the data provides relavant information about the patient.
# In this dataset stroke is a our TARGET column
# The data contains 5110 observations with 12 attributes


# # here read the data 

# In[3]:


# here read the data 

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df


# In[4]:


# df.columns


# In[5]:


df.shape


# In[6]:


df["work_type"].value_counts()


# In[7]:


# Target variable

df["stroke"].value_counts()


# In[8]:


id=df["id"]
id


# In[9]:


df=df.drop(["id"],axis=1)


# In[10]:


# here using info method we print the summary of the dataframe
# it gives quick overview of dataset

df.info()


# In[11]:


# using describe method we calculate some satistical data or analyzes numeric data

df.describe()


# In[12]:


# using isna function we find missing values in data 

df.isna().sum()


# In[13]:


# using dublicate function we find dublicate values

df.duplicated().sum()


# In[14]:


# using median method we impute the the all null values grom deta

df.fillna(df.median(), inplace=True)


# In[15]:


df.isna().sum()


# In[16]:


df.dtypes


# # Dtype converssion

# In[17]:


# using LabelEncoder we transform non numerical value into numerical values

from sklearn.preprocessing import LabelEncoder

# create an instance of LabelEncoder
label_encoder = LabelEncoder()

df.gender = label_encoder.fit_transform(df.gender)
df.age = label_encoder.fit_transform(df.age)
df.ever_married = label_encoder.fit_transform(df.ever_married)
df.work_type = label_encoder.fit_transform(df.work_type)
df.Residence_type = label_encoder.fit_transform(df.Residence_type)
df.avg_glucose_level = label_encoder.fit_transform(df.avg_glucose_level)
df.bmi  = label_encoder.fit_transform(df.bmi)
df.smoking_status = label_encoder.fit_transform(df.smoking_status)
 
#cat_cols = ['gender', 'age', 'hypertension', 'heart_disease','avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
#for col in cat_cols :
    #df[col] = label_encoder.fit_transform(df[col].astype(str))


# In[18]:


df.dtypes


#  # visualization

# In[19]:



plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot = True)
plt.show


# In[20]:


# 1 if the patient had a stroke or 0 if not
# 1 = patient had stroke
# 0 = not stroke

# using count function here we display the all categorical observation in form of graph

# 1 gender has Female=0 & Male=1, in female near 200 female had a stroke & in male near 150 male had a stroke 
# compare both then female patient had more stroke than male patients

categorical_cols = ['gender'] #'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    sns.countplot(x=col, hue='stroke', data=df)
    plt.title(col)
    plt.show()


# gender: There are more female than male patients, and a very small number of patients identify as Other

# In[21]:


categorical_cols = ['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    sns.countplot(x=col, hue='stroke', data=df)
    plt.title(col)
    plt.show()


# hypertension: The majority of patients do not have hypertension. 
# 
# heart_disease: The majority of patients do not have heart disease. 
# 
# ever_married: Most of the patients have been married at least once. 
# 
# work_type: Most of the patients are in the Private work category. There are also significant numbers in Self-employed and children. The categories Govt_job and Never_worked have fewer patients
# 
# Residence_type: The number of patients living in urban and rural areas is almost equal. 
# 
# smoking_status: Most of the patients have never smoked. The categories formerly smoked and smokes have fewer patients. There's a significant portion of patients with Unknown smoking status

# In[22]:


numerical_cols=['age', 'avg_glucose_level', 'bmi']
for col in numerical_cols:
    sns.histplot(df[col])
    plt.title(col)
    plt.show()


# age: The age of the patients varies from young to old, with the majority of patients being in the range of 40-80 years.
# 
# avg_glucose_level: Most patients have an average glucose level in the range of 50-125, but there are also many patients with higher levels. The distribution is right-skewed. 
# 
# bmi: The majority of patients have a BMI in the range of 20-40, which is considered normal to overweight. There are some outliers with extremely high BMI values. 

# In[23]:


categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    sns.histplot(x=col, hue='stroke', data=df)
    plt.title(col)
    plt.show()


# # prepare base models

# In[24]:


X = df.drop(columns="stroke")
Y = df["stroke"]


# In[25]:


X.shape, Y.shape


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

X_train.shape, X_test.shape


# In[27]:


Y_train.shape, Y_test.shape


# In[28]:


from sklearn.metrics import accuracy_score, confusion_matrix           
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model=LogisticRegression()
model.fit(X,Y)


# In[29]:


Pre = model.predict(X_test)
Pre


# In[30]:


#p0 = model.predict_proba(X_test)
#p0


# In[31]:


T_df = pd.DataFrame(Pre,columns=["stroke"])
T_df


# # 1 LR 

# In[32]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)


# In[33]:


Pred0 = model.predict(X_test)
Acc0 = accuracy_score(Y_test,Pred0)


# In[34]:


Acc0


# In[35]:


CM0 = confusion_matrix(Y_test,Pred0)
CM0


# In[36]:


(1463+0)/(1463+1+69+0)


# In[37]:


print(classification_report(Pred0,Y_test))


# # 2 Naive_bayes

# In[38]:


from sklearn.naive_bayes import GaussianNB
Model = GaussianNB()
Model.fit(X_train, Y_train)
Pred1 = Model.predict(X_test)
Acc1 = accuracy_score(Y_test, Pred1)
Acc1


# In[39]:


CM1 = confusion_matrix(Y_test,Pred1)
CM1


# In[40]:


(1310+34)/(1310+154+35+34)


# In[41]:


print(classification_report(Pred1,Y_test))


# # 3 DT 

# In[42]:


from sklearn.tree import DecisionTreeClassifier
Model0 = DecisionTreeClassifier()
Model0.fit(X_train, Y_train)
Pred2 = Model0.predict(X_test)
Acc2 = accuracy_score(Y_test, Pred2)
Acc2


# In[43]:


CM2 = confusion_matrix(Y_test,Pred2)
CM2


# In[44]:


(1385+10)/(1385+79+59+10)


# In[45]:


print(classification_report(Pred2,Y_test))


# # 4 SVM

# In[46]:


from sklearn.svm import SVC
Model2 = SVC()
Model2.fit(X_train, Y_train)
Pred3 = Model2.predict(X_test)
Acc3 = accuracy_score(Y_test, Pred3)
Acc3


# In[47]:


CM3 = confusion_matrix(Y_test,Pred3)
CM3


# In[48]:


(1464+0)/(1464+0+69+0)


# In[49]:


print(classification_report(Pred3,Y_test))


# # 5 RF

# In[50]:


from sklearn.ensemble import RandomForestClassifier
Model3 = RandomForestClassifier()
Model3.fit(X_train, Y_train)
Pred4 = Model3.predict(X_test)
Acc4 = accuracy_score(Y_test, Pred4)
Acc4


# In[51]:


CM4 = confusion_matrix(Y_test,Pred4)
CM4


# In[52]:


(1461+0)/(1461+3+69+0)


# In[53]:


print(classification_report(Pred4,Y_test))


# # 6 KNN

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
Model4 = KNeighborsClassifier()
Model4.fit(X_train, Y_train)
Pred5 = Model4.predict(X_test)
Acc5 = accuracy_score(Y_test,Pred5)
Acc5


# In[55]:


CM5 = confusion_matrix(Y_test,Pred5)
CM5


# In[56]:


(1457+0)/(1457+7+69+0)


# In[57]:


print(classification_report(Pred5,Y_test))


# # 7 XGboost 

# In[58]:


xgb_model_1 = xgb.XGBClassifier(objective="binary:logistic",random_state=42)


# In[59]:


xgb_model_1.fit(X,Y)


# In[60]:


pred_xgb= xgb_model_1.predict(X)
pred_xgb


# In[61]:


acc_xgb = accuracy_score(Y,pred_xgb)
acc_xgb


# In[62]:


cm_xgb = confusion_matrix(Y,pred_xgb)
cm_xgb


# In[63]:


(4861+209)/(4861+0+40+209)


# In[64]:


print(classification_report(pred_xgb,Y))


# # 8 Adaboost classifier

# In[65]:


Ada_model_1 = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada_model_1.fit(X,Y)


# In[66]:


pred_ada = Ada_model_1.predict(X)
pred_ada


# In[67]:


acc_ada = accuracy_score(Y,pred_ada)
acc_ada


# In[68]:


cm_ada = confusion_matrix(Y,pred_ada)
cm_ada


# In[69]:


(4856+5)/(4856+5+5+244)


# In[70]:


print(classification_report(pred_ada,Y))


# In[ ]:





# In[71]:


DATA = [["LR",0.954337899543379,1.00,0.03,0.96,0.29,0.98,0.05], ["N_B",0.8767123287671232,0.89,0.49,0.97,0.18,0.93,0.26], ["DT",0.9073711676451403,0.94,0.17,0.96,0.12,0.95,0.14],["SVM",0.9549902152641878,1.00,0.00,0.95,0.00,0.98,0.00], ["RF",0.9530332681017613,1.00,0.00,0.95,0.00,0.98,0.00], ["KNN",0.9504240052185258,0.99,0.00,0.95,0.00,0.97,0.00],["XGboost",0.9921722113502935,1.00,0.84,0.99,1.00,1.00,0.91],["Adabosst",0.9512720156555773,1.00,0.02,0.95,0.50,0.98,0.04]]
DF = pd.DataFrame(DATA, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
DF


# In[72]:


#plt.figure(figsize=(8,6))
#sns.heatmap(df.corr(), annot = True)
#plt.show


# # EDA

# # Outliers Tratement

# In[73]:


df.boxplot()


# In[74]:


# using IQR method we find outliers

df_numeric = df.select_dtypes(include = [np.number])

q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1

lb = q1 - 1.5 *iqr
ub = q3 + 1.5*iqr

outliers = df_numeric[(df_numeric<lb) | (df_numeric > ub)]


# In[75]:


outliers.count()


# In[76]:


# check outliers for numerical columns

# df.boxplot("age")


# In[77]:


df.age.hist()


# In[78]:


sns.distplot(df.age)   


# In[ ]:





# In[79]:


# df.boxplot("avg_glucose_level")


# In[80]:


df.avg_glucose_level.hist()


# In[81]:


sns.distplot(df.avg_glucose_level)


# In[ ]:





# In[82]:


df.boxplot("bmi")


# In[83]:


df.bmi.hist()


# In[84]:


sns.distplot(df.bmi)


# In[85]:


for col in df_numeric.columns:
    col_median = df_numeric[col].median()
    df_numeric.loc[df_numeric[col] < lb[col], col] = col_median
    df_numeric.loc[df_numeric[col] > ub[col], col] = col_median


# In[86]:


outliers.ai = df_numeric[(df_numeric<lb) | (df_numeric > ub)]


# In[87]:


outliers.ai.count()


# In[ ]:





# # Skewness treatment
# If less than -1 or greater than 1, the distribution is highly skewed. 
# 
# If between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed. 
# 
# If between -0.5 and 0.5, the distribution is approximately symmetric

# In[88]:


df.skew()


# # prepare all the model after outlier treatment

# In[89]:


x = df.drop(columns="stroke")
y = df["stroke"]


# In[90]:


x


# In[91]:


y


# In[92]:


x.shape, y.shape


# In[ ]:





# In[93]:


#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

X_train.shape, X_test.shape


# In[94]:


y_train.shape, y_test.shape


# In[95]:


#from sklearn.metrics import accuracy_score, confusion_matrix           
#from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x,y)


# In[96]:


pred = model.predict(X_test)
pred


# In[97]:


p1=model.predict_proba(X_test)
p1


# In[98]:


df1 = pd.DataFrame(pred,columns=["stroke"])


# In[99]:


df1


# # 1 Logistic Regrassion

# In[100]:


from sklearn.linear_model import LogisticRegression
MODEL = LogisticRegression()
MODEL.fit(X_train,y_train)


# In[101]:


PRED = MODEL.predict(X_test)
ACC = accuracy_score(y_test,PRED)


# In[102]:


ACC


# In[103]:


CM = confusion_matrix(y_test,PRED)
CM


# In[104]:


(1458+2)/(1458+5+68+2)


# In[105]:


from sklearn.metrics import classification_report


# In[106]:


print(classification_report(PRED,y_test))


# # 2 Naive_bayes

# In[107]:


from sklearn.naive_bayes import GaussianNB
MODEL1= GaussianNB()
MODEL1.fit(X_train, y_train)
PRED1 = MODEL1.predict(X_test)
ACC1 = accuracy_score(y_test, PRED1)
ACC1


# In[108]:


CM1 = confusion_matrix(y_test,PRED1)
CM1


# In[109]:


(1303+35)/(1303+160+35+35)


# In[110]:


print(classification_report(PRED1,y_test))


# # 3 Decision tree

# In[111]:


from sklearn.tree import DecisionTreeClassifier
MODEL2 = DecisionTreeClassifier()
MODEL2.fit(X_train, y_train)
PRED2 = MODEL2.predict(X_test)
ACC2 = accuracy_score(y_test, PRED2)
ACC2


# In[112]:


CM2 = confusion_matrix(y_test,PRED2)
CM2


# In[113]:


(1398+11)/(1398+65+59+11)


# In[114]:


print(classification_report(PRED2,y_test))


# # 4 SVM

# In[115]:


from sklearn.svm import SVC
MODEL3 = SVC()
MODEL3.fit(X_train, y_train)
PRED3 = MODEL3.predict(X_test)
ACC3 = accuracy_score(y_test, PRED3)
ACC3


# In[116]:


CM3 = confusion_matrix(y_test, PRED3)
CM3


# In[117]:


(1463+0)/(1463+0+70+0)


# In[118]:


print(classification_report(PRED3,y_test))


# In[ ]:





# # 5 Random Forest

# In[119]:


from sklearn.ensemble import RandomForestClassifier
MODEL4 = RandomForestClassifier()
MODEL4.fit(X_train, y_train)
PRED4= MODEL4.predict(X_test)
ACC4 = accuracy_score(y_test, PRED4)
ACC4


# In[120]:


CM4 = confusion_matrix(y_test,PRED4)
CM4


# In[121]:


(1461+0)/(1461+2+70+0)


# In[122]:


print(classification_report(PRED4,y_test))


# # 6 KNN

# In[123]:


from sklearn.neighbors import KNeighborsClassifier
MODEL5 = KNeighborsClassifier()
MODEL5.fit(X_train, y_train)
PRED5 = MODEL5.predict(X_test)
ACC5 = accuracy_score(y_test,PRED5)
ACC5


# In[124]:


CM5 = confusion_matrix(y_test,PRED5)
CM5


# In[125]:


(1451+3)/(1451+12+67+3)


# In[126]:


print(classification_report(PRED5,y_test))


# # 7 XGboost

# In[127]:


xgb_model_2 = xgb.XGBClassifier(objective="binary:logistic",random_state=42)
xgb_model_2.fit(X,Y)


# In[128]:


pred_xgb_2= xgb_model_2.predict(X)
pred_xgb_2


# In[129]:


cm_xgb_2 = confusion_matrix(Y,pred_xgb_2)
cm_xgb_2


# In[130]:


acc_xgb_2 = accuracy_score(Y,pred_xgb_2)
acc_xgb_2


# In[131]:


print(classification_report(pred_xgb_2,Y))


# # 8 Adaboost

# In[132]:


Ada_model_2 = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada_model_2.fit(X,Y)


# In[133]:


pred_ada_2 = Ada_model_2.predict(X)
pred_ada_2


# In[134]:


acc_ada_2 = accuracy_score(Y,pred_ada_2)
acc_ada_2


# In[135]:


cm_ada_2 = confusion_matrix(Y,pred_ada_2)
cm_ada_2


# In[136]:


print(classification_report(pred_ada_2,Y))


# # Dataframe after outliers treatment

# In[137]:


data_eda = [["LR",0.9523809523809523,1.00,0.03,0.96,0.29,0.98,0.05 ],["N_B",0.87279843444227,0.89,0.50,0.97,0.18,0.93,0.26],["DT",0.9191128506196999,0.96,0.16,0.96,0.14,0.96,0.14],["SVM",0.954337899543379,1.00,0.00,0.95,0.00,0.98,0.00],["RF",0.9530332681017613,1.00,0.00,0.95,0.00,0.98,0.00],["KNN",0.9484670580560991,0.99,0.04,0.96,0.20,0.97,0.07],["XGboost",0.9921722113502935,1.00,0.84,0.99,1.00,1.00,0.91],["Adaboost",0.9512720156555773,1.00,0.02,0.95,0.50,0.98,0.04]]
df_eda = pd.DataFrame(data_eda, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
df_eda


# # Dataframe of basic model

# In[138]:


DATA = [["LR",0.954337899543379,1.00,0.03,0.96,0.29,0.98,0.05], ["N_B",0.8767123287671232,0.89,0.49,0.97,0.18,0.93,0.26], ["DT",0.9073711676451403,0.94,0.17,0.96,0.12,0.95,0.14],["SVM",0.9549902152641878,1.00,0.00,0.95,0.00,0.98,0.00], ["RF",0.9530332681017613,1.00,0.00,0.95,0.00,0.98,0.00], ["KNN",0.9504240052185258,0.99,0.00,0.95,0.00,0.97,0.00],["XGboost",0.9921722113502935,1.00,0.84,0.99,1.00,1.00,0.91],["Adabosst",0.9512720156555773,1.00,0.02,0.95,0.50,0.98,0.04]]
DF = pd.DataFrame(DATA, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
DF


# # Data Imbalance techniques using IMBlearn

# # IMBlearn with Oversampling

# In[139]:


import imblearn
from imblearn import under_sampling, over_sampling


# In[140]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
x1, y1 = ros.fit_resample(x,y)


# In[141]:


x1.shape, y1.shape


# In[142]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=100)
x1_train.shape, x1_test.shape


# In[143]:


y1_train.shape, y1_test.shape


# # 1 Logistic regrassion with oversampling

# In[144]:


from sklearn.linear_model import LogisticRegression
model_O_1 = LogisticRegression()
model_O_1.fit(x1_train, y1_train)


# In[145]:


pred_O_1= model_O_1.predict(x1_test)
pred_O_1
acc_O_1 = accuracy_score(y1_test,pred_O_1)


# In[146]:


acc_O_1


# In[147]:


cm_O_1 = confusion_matrix(y1_test,pred_O_1)
cm_O_1


# In[148]:


(1047+1197)/(1047+391+281+1197)


# In[149]:


print(classification_report(pred_O_1,y1_test))


# # 2  Naive_bayes

# In[150]:


from sklearn.naive_bayes import GaussianNB
model_O_2 = GaussianNB()
model_O_2.fit(x1_train,y1_train)


# In[151]:


pred_O_2 = model_O_2.predict(x1_test)
pred_O_2


# In[152]:


acc_O_2 = accuracy_score(y1_test,pred_O_2)


# In[153]:


acc_O_2


# In[154]:


cm_O_2 = confusion_matrix(y1_test, pred_O_2)
cm_O_2


# In[155]:


(1007+1208)/(1007+431+270+1208)


# In[156]:


print(classification_report(pred_O_2,y1_test))


# # 3 DT with oversampling

# In[157]:


from sklearn.tree import DecisionTreeClassifier
model_O_3 = DecisionTreeClassifier()
model_O_3.fit(x1_train, y1_train)
pred_O_3 = model_O_3.predict(x1_test)
acc_O_3 = accuracy_score(y1_test, pred_O_3)
acc_O_3


# In[158]:


cm_O_3 = confusion_matrix(y1_test,pred_O_3)
cm_O_3


# In[159]:


(1361+1478)/(1361+77+0+1478)


# In[160]:


print(classification_report(pred_O_3,y1_test))


# # 4 SVM with Oversampling

# In[161]:


from sklearn.svm import SVC
model_O_4 = SVC()
model_O_4.fit(x1_train, y1_train)
pred_O_4 = model_O_4.predict(x1_test)
acc_O_4 = accuracy_score(y1_test, pred_O_4)
acc_O_4


# In[162]:


cm_O_4 = confusion_matrix(y1_test,pred_O_4)
cm_O_4


# In[163]:


(1160+666)/(1160+278+812+666)


# In[164]:


print(classification_report(pred_O_4,y1_test))


# # 5 RF with Oversampling

# In[165]:


from sklearn.ensemble import RandomForestClassifier
model_O_5 = RandomForestClassifier()
model_O_5.fit(x1_train, y1_train)
pred_O_5 = model_O_5.predict(x1_test)
acc_O_5 = accuracy_score(y1_test,pred_O_5)
acc_O_5


# In[166]:


cm_O_5 =confusion_matrix(y1_test,pred_O_5)
cm_O_5


# In[167]:


(1406+1478)/(1406+32+0+1478)


# In[168]:


print(classification_report(pred_O_5,y1_test))


# In[ ]:





# In[ ]:





# # 6 KNN with Oversampling

# In[169]:


from sklearn.neighbors import KNeighborsClassifier
model_O_6 = KNeighborsClassifier()
model_O_6.fit(x1_train, y1_train)
pred_O_6 = model_O_6.predict(x1_test)
acc_O_6 = accuracy_score(y1_test,pred_O_6)
acc_O_6


# In[170]:


cm_O_6 =confusion_matrix(y1_test,pred_O_6)
cm_O_6


# In[171]:


(1224+1478)/(1224+214+0+1478)


# In[172]:


print(classification_report(pred_O_6,y1_test))


# In[173]:


data_over = [["LR_IMB_O",0.7695473251028807,0.73,0.81,0.79,0.75,0.76,0.78], ["N_B_IMB_O",0.75960219478738,0.70,0.82,0.79,0.74,0.74,0.78], ["DT_IMB_O",0.9735939643347051,0.95,1.00,0.95,1.00,0.97,0.97],["SVM_IMB_O",0.6262002743484225,0.81,0.45,0.59,0.71,0.68,0.55], ["RF_IMB_O",0.9890260631001372,0.98,1.00,1.00,0.98,0.99,0.99], ["KNN_IMB_O",0.9266117969821673,0.85,1.00,1.00,0.87,0.92,0.93]]
df_over = pd.DataFrame(data_over, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
df_over


# In[ ]:





# # IMBlearn with Undersampling

# In[174]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
x2, y2 = rus.fit_resample(x,y)


# In[175]:


x2.shape,y2.shape


# In[176]:


x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=100)

x2_train.shape, x2_test.shape


# In[177]:


y2_train.shape, y2_test.shape


# # 1 LR with Undersampling

# In[178]:


from sklearn.linear_model import LogisticRegression
model_U_1 = LogisticRegression()
model_U_1.fit(x2_train,y2_train)


# In[179]:


pred_U_1 = model_U_1.predict(x2_test)
pred_U_1


# In[180]:


acc_U_1= accuracy_score(y2_test,pred_U_1)
acc_U_1


# In[181]:


cm_U_1=confusion_matrix(y2_test,pred_U_1)
cm_U_1


# In[182]:


(52+64)/(52+21+13+64)


# In[183]:


print(classification_report(pred_U_1,y2_test))


# # 2 Naive_bayes

# In[184]:


from sklearn.naive_bayes import GaussianNB
model_U_2 = GaussianNB()
model_U_2.fit(x2_train,y2_train)
pred_U_2 = model_U_2.predict(x2_test)
acc_U_2 = accuracy_score(y2_test,pred_U_2)
acc_U_2


# In[185]:


cm_U_2 =confusion_matrix(y2_test, pred_U_2)
cm_U_2


# In[186]:


(53+63)/(53+20+14+63)


# In[187]:


print(classification_report(pred_U_2,y2_test))


# # 3 DT with Undersampling

# In[188]:


from sklearn.tree import DecisionTreeClassifier
model_U_3 = DecisionTreeClassifier()
model_U_3.fit(x2_train, y2_train)
pred_U_3 = model_U_3.predict(x2_test)
acc_U_3 = accuracy_score(y2_test, pred_U_3)
acc_U_3


# In[189]:


cm_U_3 =confusion_matrix(y2_test, pred_U_3)
cm_U_3


# In[190]:


(44+57)/(44+29+20+57)


# In[191]:


print(classification_report(pred_U_3,y2_test))


# # 4 SVM with Undersampling

# In[192]:


from sklearn.svm import SVC
model_U_4 = SVC()
model_U_4.fit(x2_train, y2_train)
pred_U_4 = model_U_4.predict(x2_test)
acc_U_4 = accuracy_score(y2_test, pred_U_4)
acc_U_4


# In[193]:


cm_U_4 =confusion_matrix(y2_test, pred_U_4)
cm_U_4


# In[194]:


(59+28)/(59+14+49+28)


# In[195]:


print(classification_report(pred_U_4,y2_test))


# # 5 RF with Undersampling

# In[196]:


from sklearn.ensemble import RandomForestClassifier
model_U_5 = RandomForestClassifier()
model_U_5.fit(x2_train, y2_train)
pred_U_5 = model_U_5.predict(x2_test)
acc_U_5 = accuracy_score(y2_test,pred_U_5)
acc_U_5


# In[197]:


cm_U_5 =confusion_matrix(y2_test, pred_U_5)
cm_U_5


# In[198]:


(48+64)/(48+25+13+64)


# In[199]:


print(classification_report(pred_U_5,y2_test))


# # 6 KNN with Undersampling

# In[200]:


from sklearn.neighbors import KNeighborsClassifier
model_U_6 = KNeighborsClassifier()
model_U_6.fit(x2_train, y2_train)
pred_U_6 = model_U_6.predict(x2_test)
acc_U_6 = accuracy_score(y2_test,pred_U_6)
acc_U_6


# In[201]:


cm_U_6 =confusion_matrix(y2_test, pred_U_6)
cm_U_6


# In[202]:


(50+57)/(50+23+20+57)


# In[203]:


print(classification_report(pred_U_6,y2_test))


# # model after Undersampling

# In[204]:


data = [["LR_IMB_U",0.7733333333333333,0.71,0.83,0.80,0.75,0.75,0.79], ["N_B_IMB_U",0.7733333333333333,0.73,0.82,0.76,0.79,0.76,0.79], ["DT_IMB_O",0.6733333333333333,0.60,0.74,0.69,0.66,0.64,0.69],["SVM_IMB_U",0.58,0.81,0.36,0.55,0.67,0.65,0.47], ["RF_IMB_U",0.7466666666666667,0.66,0.87,0.83,0.73,0.73,0.79], ["KNN_IMB_U",0.7133333333333334,0.68,0.74,0.71,0.71,0.70,0.73]]
df = pd.DataFrame(data, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
df


# # model after Oversampling

# In[205]:


data_over = [["LR_IMB_O",0.7695473251028807,0.73,0.81,0.79,0.75,0.76,0.78], ["N_B_IMB_O",0.75960219478738,0.70,0.82,0.79,0.74,0.74,0.78], ["DT_IMB_O",0.9735939643347051,0.95,1.00,0.95,1.00,0.97,0.97],["SVM_IMB_O",0.6262002743484225,0.81,0.45,0.59,0.71,0.68,0.55], ["RF_IMB_O",0.9890260631001372,0.98,1.00,1.00,0.98,0.99,0.99], ["KNN_IMB_O",0.9266117969821673,0.85,1.00,1.00,0.87,0.92,0.93]]
df_over = pd.DataFrame(data_over, columns=["Algo","Acc","precision_0","precision_1","recall_0","recall_1","f1-score_0","f1-score_1"])
df_over


# # XGBboost classifier

# In[206]:


# !pip install xgboost
import xgboost as xgb


# In[207]:


from xgboost import XGBClassifier


# In[208]:


xgb_model=xgb.XGBClassifier(objective="binary:logistic",random_state=42)


# In[209]:


xgb_model.fit(x,y)


# In[210]:


pred_3= xgb_model.predict(x)
pred_3


# In[211]:


cm_3 = confusion_matrix(y,pred_3)
cm_3


# In[212]:


acc_3 = accuracy_score(y,pred_3)
acc_3


# In[213]:


print(classification_report(pred_3,y))


# # Adaboost classifier

# In[214]:


from sklearn.ensemble import AdaBoostClassifier
Ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada_model.fit(x,y)


# In[215]:


pred_4= Ada_model.predict(x)
pred_4


# In[216]:


acc_4 = accuracy_score(y,pred_4)
acc_4


# In[217]:


cm_4 = confusion_matrix(y,pred_4)
cm_4


# In[218]:


(4853+6)/(4853+6+242+8)


# In[219]:


print(classification_report(pred_4,y))


# In[220]:


data = [["XGBboost",0.9929522317932654],["Adaboost",0.9510667449598748]]
df = pd.DataFrame(data, columns=["Algo","Acc"])
df


# # Feature selection

# In[221]:


fe_x = X
fe_y = Y


# In[222]:


fe_x


# In[223]:


fe_y


# In[224]:


fe_X_train, fe_X_test, fe_y_train, fe_y_test = train_test_split(fe_x, fe_y, test_size = 0.3, random_state = 100)

fe_X_train.shape, fe_X_test.shape


# In[225]:


fe_y_train.shape, fe_y_test.shape


# # RFE for RF

# In[226]:


Model_fe = RandomForestClassifier()
Model_fe.fit(fe_X_train, fe_y_train)
Pred_fe= Model_fe.predict(fe_X_test)
fe_ACC = accuracy_score(fe_y_test, Pred_fe)
fe_ACC


# In[227]:


fe_CM = confusion_matrix(fe_y_test,Pred_fe)
fe_CM


# In[228]:


(1461+0)/(1461+2+70+0)


# In[229]:


print(classification_report(fe_y_test, Pred_fe))


# # RFE

# In[230]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()


# In[231]:


rfe=RFE(dtree)


# In[232]:


rfe.fit(fe_X_train, fe_y_train)


# In[233]:


rfe.support_


# In[234]:


rfe.ranking_


# In[235]:


fe_x1 = X
fe_y1 = Y


# In[236]:


fe_x1_train,fe_x1_test,fe_y1_train,fe_y1_test = train_test_split(fe_x1,fe_y1,test_size=0.2)


# In[237]:


fe_x1_train.shape, fe_x1_test.shape


# In[238]:


fe_y1_train.shape, fe_y1_test.shape


# In[239]:


model_df_rf = RandomForestClassifier()


# In[240]:


model_df_rf.fit(fe_x1_train, fe_y1_train)


# In[241]:


pred_df_rf = model_df_rf.predict(fe_x1_test)


# In[242]:


acc_df_rf = accuracy_score(fe_y1_test,pred_df_rf)
acc_df_rf


# In[243]:


cm_df_rf = confusion_matrix(fe_y1_test, pred_df_rf)
cm_df_rf


# In[249]:


(960+0)/(960+1+61+0)


# In[250]:


print(classification_report(fe_y1_test, pred_df_rf))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[246]:


Ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)
Ada_model.fit(x,y)
pred_4= Ada_model.predict(x)
pred_4
acc_4 = accuracy_score(y,pred_4)
acc_4
cm_4 = confusion_matrix(y,pred_4)
cm_4
print(classification_report(pred_4,y))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[247]:


df.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[248]:


xgb_model=xgb.XGBClassifier(objective="binary:logistic",random_state=42)
xgb_model.fit(x,y)
pred_3= xgb_model.predict(x)
pred_3
cm_3 = confusion_matrix(y,pred_3)
cm_3
acc_3 = accuracy_score(y,pred_3)
acc_3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




