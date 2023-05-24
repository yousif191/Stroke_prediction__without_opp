import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score 
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


Data=pd.read_csv('.archhealthcare-dataset-stroke-data.csv')
        

print(f'The Data :\n{Data}') 
print(f'\nData size={Data.shape}\n')
print(f'\nthe columns of data: \n{Data.columns}')
print(f'\nFirst 5 Rows in from the Data:\n {Data.head()}')
print(f'The Data Info : {Data.info()}')
print(f'\nNumber of duplicated row={Data.duplicated().sum()}')
print(f'The empty cell in each columns before cleaning:\n{Data.isnull().sum()}')
print(f'The Describe of Data :\n {Data.describe()}')
print(f'The Element of Gender column without Repeating:\n{Data.gender.unique()}')
        

        

bmi_mean=Data['bmi'].mean()
Data['bmi']=Data['bmi'].fillna(bmi_mean)
Data['gender']=Data['gender'].replace({'Other': 'Male'})
print(f'The empty cell in each columns after cleaning:\n{Data.isnull().sum()}')
        

plt.figure(figsize=(10, 5))
sns.histplot(Data["age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x="gender",data=Data)
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="smoking_status",y="bmi",data=Data)
plt.title("BMI Distribution by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("BMI")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x="stroke",data=Data)
mean_stroke =Data["stroke"].mean()
plt.axhline(mean_stroke, color='red',linestyle='--',label='Mean')
plt.show()

plt.figure(figsize=(8, 8))
Data["smoking_status"].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Smoking Status Distribution")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x="age", y="avg_glucose_level",data=Data)
plt.title("Age vs. Average Glucose Level")
plt.xlabel("Age")
plt.ylabel("Average Glucose Level")
plt.show()
        
        
        

Data_types = Data.dtypes
for i in range(Data.shape[1]):
        if Data_types[i] == "O":
             pr_data = preprocessing.LabelEncoder()
             Data[Data.columns[i]]=pr_data.fit_transform(Data[Data.columns[i]])

                
                
plt.figure(figsize=(10, 8))
sns.heatmap(Data.corr(),annot=True)
plt.title("Correlation Heatmap")
plt.show()
        
plt.figure(figsize=(10, 5))
sns.boxplot(x="ever_married", y="age",data=Data)
plt.title("Age Distribution by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Age")
plt.show()
        

Data =Data.drop(['work_type','id'], axis=1)
print(f'the columns of data:\n{Data.columns}')
print(f'\nSize of data after droping={Data.shape}')
        

    

features=Data.iloc[:, :-1]
scaler=preprocessing.MinMaxScaler()
scaled_data=scaler.fit_transform(features)
scaled_data=pd.DataFrame(scaled_data, columns=features.columns)
print(f'Data after scaling :\n{scaled_data} ')
Y=Data.iloc[:, -1]
        

X_train,X_test,y_train,y_test=train_test_split(scaled_data,Y,test_size=0.2)
print(f'X_train=\n{X_train}')
print(f'X_test=\n{X_test}')
print(f'Y_train=\n{y_train}')
print(f'Y_test=\n{y_test}')
        
        
        

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
        
con = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for Logistic Regression:\n',con)
sns.heatmap(con,annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap of Confusion Matrix for Logistic Regression")
plt.show()
        
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy for Logistic Regression:{accuracy}')
        
f1 = f1_score(y_test, y_pred,average='micro')
print(f'F1 Score for Logistic Regression: {f1}')
        
recall = recall_score(y_test,y_pred,average='micro')
print(f'Recall Score for Logistic Regression:{recall}')
        
      
        

clf=SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred_svc=clf.predict(X_test)
        
con_svc=confusion_matrix(y_test, y_pred_svc)
print('Confusion Matrix for SVC:\n', con_svc)
sns.heatmap(con_svc, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Confusion Matrix for SVC")
plt.show()
        
accuracy_svc=accuracy_score(y_test, y_pred_svc)
print(f'Accuracy for SVC: {accuracy_svc}')
        
f1_svc=f1_score(y_test, y_pred_svc, average='micro')
print(f'F1 Score for SVC: {f1_svc}')
        
recall_svc=recall_score(y_test, y_pred_svc, average='micro')
print(f'Recall Score for SVC: {recall_svc}')
    
    
    

    
      

