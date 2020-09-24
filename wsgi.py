#!/usr/bin/env python
# coding: utf-8

# #### Week 8 - Functions

# In[1]:


# Import Libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add in Week 3
# for graphical output to df.head and df.describe from within a function, use display NOT print
from IPython.display import display 

# Feature Engineering
from sklearn.preprocessing import MinMaxScaler

# Model and Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,recall_score


# In[2]:


# Run environment Setup
import warnings
warnings.simplefilter("ignore")

#  numpy print options
# used fixed dpoint notation for floats with 4 decimals
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# Display options on terminal for pandas dataframes
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# global variable is available to all functions in this python file
TRAINED_MODEL = 0
TO_SCALE = []
SCALER = 0
TO_DROP = []


# #### Read data

# In[3]:


def read_data(filename):
    print("\n*****FUNCTION read_data*****")
    
    # Read the data file into a df
    df = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'],filename))
        
    # See the data in the df
    display(df.head())   
    
    # Full data set Shape
    print("Shape of Full set:", df.shape)
        
    # Keep TRAIN set only , 40000 rows , 50% churn
    df= df[df.traintest == 1]
    
    # Train data set Shape
    print("Shape of Train set:", df.shape)
    
    return(df)

# end of function read_data


# #### Data Exploration

# In[4]:


def disp_df_info(df):
    print("\n*****FUNCTION disp_df_info*****")
    
    # Create a Pie Chart to check Balance
    df['churn'].value_counts(sort=True)
    
    #Plotting Parameters
    plt.figure(figsize=(5,5))
    sizes = df['churn'].value_counts(sort=True)
    colors = ["grey", 'purple']
    labels = ['No', 'Yes']

    #Plot Pie Chart
    plt.pie(sizes, colors = colors, labels = labels, autopct='%1.1f%%', shadow=True, startangle=270,)
    plt.title('Percentage of Churn in Dataset')
    plt.show()
           
    # display column Headers
    print("Column Headers:")
    print(df.columns,)
    
      
    # print first 10 data samples
    print("Top 10 rows:")
    display(df.head(10))
    
    # Describe the df to check if features need scaling
    print("Statistics:")
    display(df.describe())
    
    # Identify the Categorical Vars and identify nulls
    print("Information:")
    print(df.info())
     
    # Count Nulls 
    print("Null Count:")
    print(df.isnull().sum())
    
    # Percent of Nulls
    print("Null Percent:") 
    print(df.isnull().mean())
    
# end of function disp_df_info


# #### Data Cleaning

# In[5]:


def data_cleaning(df_input):
    print("\n*****FUNCTION data_cleaning*****")
    
    df = df_input.copy(deep=True)
    
    # Print Shape
    print("\nShape Before Dropping rows and columns:", df.shape)
       
    # Drop unwanted columns
    df.drop(['Unnamed: 0', 'X', 'customer', 'traintest','churndep'],axis=1,inplace=True)
    display(df.head())
    
    # Drop rows with Nulls, using df.dropna(), less than 3 % drop is OK
    df = df.dropna()
    
    # Drop rows with Nulls, using df.dropna(), less than 3 % drop is OK
    print("\nShape After Dropping rows and Columns:", df.shape)
    
    return(df)   
# end of functiom clean_data    


# #### Data Split into X/Feature and Y/target

# In[6]:


def data_split(df_input):
    print("\n*****FUNCTION data_split*****")
    
    df = df_input.copy(deep=True)
    
    # Create Y var
    y = df['churn']
    print('Y/Target Var:')
    display(y.head(10))

    # Create X var
    x = df.drop(['churn'], axis=1)
    print('X/Feature Var:')
    display(x.head(10))
    
    return(x,y)
# end of function data_split


# #### Feature Engineering

# In[7]:


def feature_engineering(x_input):
    print("\n*****FUNCTION feature_engineering*****")

       
    return(x)    
# end of function feature_engineering


# #### Feature Selection

# In[8]:


def feature_selection(x_input):
    print("\n*****FUNCTION feature_selection*****")
    
    x = x_input.copy(deep=True)
    
    global TO_DROP
    
    # Check the correlation of the variables
    corr_mat = x.corr()   
    
    # Correlation Matrix visualized as Heatmap
    print("\nCorrelation Martix for X/Feature Space:\n")
    plt.figure(figsize=(20,20))
    sns.heatmap(corr_mat, cmap='coolwarm', center = 0 , vmin=-1, vmax=1)
    plt.show()
    
    # Create correlation matrix
    corr_matrix = x.corr().abs()
    #print(corr_matrix)
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #print(upper)
    
    # Find index of feature columns with correlation greater than 0.75
    TO_DROP = [column for column in upper.columns if any(upper[column] > 0.75)]
    print("\nColumns to drop due to Feature Selection:\n", TO_DROP)
      
    
    # Shape before dropping features
    print('\nShape BEFORE Dropping redundant features:\n', x.shape)

    # Drop features
    x.drop(x[TO_DROP], axis=1, inplace=True)

    # Shape after dropping features
    print('\nShape AFTER Dropping redundant features:\n', x.shape)
     
    return(x)
# end of function feature_selection


# #### Feature Scaling -  For Selected columns ONLY

# In[9]:


def feature_scaling(x_input):
    print("\n*****FUNCTION feature_scaling*****")

    global SCALER 
    global TO_SCALE
    
    x = x_input.copy(deep=True)
    
    # Save the list of features after selection
    x_list = x.columns
        
    # Identify the total integer vars = numbers + ohe
    totint = [var for var in x.columns if x[var].dtype=='int64']
    print("\nTotal Integer Variables:",len(totint))
    print("\nTotal Integer Variables:",totint)
    
    # Identify the integer vars = numbers, subset of total intergers to scale are those that dont have a range of 1
    intvar = [var for var in x.columns              if (x[var].dtype=='int64' and (x[var].max()-x[var].min() != 1 ))] 
    print("\nInteger Variables:",len(intvar))
    print("\nInteger Variables:",intvar)
    
    # Identify the float vars
    contvar = [var for var in x.columns if x[var].dtype=='float64' ] 
    print("\nFloat Variables:",len(contvar))      
    print("\nFloat Variables:",contvar)
    
    # Create a list of columns to scale that includes floats and integers where the range is not 1
    # This can be be a hardcoded list of column names if there are few x vars
    TO_SCALE = intvar + contvar
    print("\nNumeric Variables to scale:",len(TO_SCALE))
    print("\nNumeric Varaibles:",TO_SCALE)
    
    
    # Create a separate df of columns to scale
    dftoscale = x[TO_SCALE]
       
    # Call scaler to scale the df
    SCALER = MinMaxScaler()
    SCALER.fit(dftoscale)
    dftoscale = SCALER.transform(dftoscale)
        
    # put the scaled back into original
    x[TO_SCALE]=dftoscale
        
    return(x)
# end of function feature_scaling


# #### Model Fitting 

# In[10]:


def build_logreg_model(x_input, y_input):
    print("\n*****FUNCTION build_logreg_model*****")
    
    x = x_input.copy(deep=True)
    y = y_input.copy(deep=True)
    
    # Call Logistic Regession with no penalty
    mod = LogisticRegression(penalty='none')
    mod.fit(x,y)
    
    # Print the Intercept and the coef
    print('Intercept:', mod.intercept_)
    print('Coefficients:', mod.coef_)
    
    # Score the model
    score = mod.score(x, y)
    print('Accuracy Score:',score)
    # probability of being 0, 1 in binary clasification , threshold is .5
    y_prob=mod.predict_proba(x)
    print('Probabilities:',y_prob)
    
    # probability converted to predictions
    y_pred = mod.predict(x)
    print('Predictions',y_pred)
    
     #### Model Metrics
    
    # Confusion Matrix gives the mistakes made by the classifier
    cm =confusion_matrix(y, y_pred)
    print('Confusion Matrix:\n',cm)
    
    # Confusion Matrix visualized
    print('Confusion Matrix Visualized:\n')
    plt.figure(figsize= (8,6))
    sns.heatmap(cm, annot= True, fmt= 'd', cmap = 'Reds')
    plt.xlabel('Predicted y_pred')
    plt.ylabel('Actuals / labels - y')
    plt.show()
    
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    
    # For Logistic Regression the model score is the Accuracy Ratio
    # (TP+TN)/(TP+TN+FP+FN)
    acc = accuracy_score(y,y_pred)
    print('Accuracy:',acc)
    
    # Precion = TP/(TP+FP)
    # Interpretation: out of all the predicted positive classes, how much we predicted correctly.
    pre = precision_score(y,y_pred)
    print('Precision:',pre)
    
    # Specificity = TN/(TN+FN)
    # Interpretation: out of all the -ve samples, how many was the classifier able to pick up
    spec = TN/(TN + FP)
    
    # Recall/Sensitivity/tpr = TP/(TP+FN)
    # Interpretation: out of all the +ve samples, how many was the classifier able to pick up
    rec = recall_score(y,y_pred)
    tpr=rec
    print('Recall:',rec)
    
    # false positive rate(fpr) = FP/(FP + TN) = 1-specificity
    # Interpretation: False alarm rate
    fpr = FP/(FP + TN)
    print('False Positive Rate',fpr)
    
    # Print Completion
    print('********Model Ready to be used/invoked******************')
    
    # return the trained model
    return(mod,score) 
# end of function build_logreg_model


# #### Week 9 - Flask

# In[11]:


# Import Flask 
from flask import Flask
from flask import render_template
from flask import request
from flask import send_file


# In[12]:


# import werkzeug to run your app as a web application
# from werkzeug.serving import run_simple


# In[13]:


# Create input file folder
upload_folder_name = 'input_capstone_folder'
upload_folder_path = os.path.join(os.getcwd(),upload_folder_name)
print('Upload folder path is:',upload_folder_path)
if not os.path.exists(upload_folder_path):
    os.mkdir(upload_folder_path)


# In[14]:


# Instantiate the Flask object 
application = Flask(__name__)
print('Flask object Instantiated')


# In[15]:


application.config['UPLOAD_FOLDER'] = upload_folder_path


# In[16]:


# home displays trainform.html
@application.route("/train", methods=['GET'])
def train():
    return render_template('trainform.html')
# end of home


# In[17]:


# submit on trainform.html
@application.route("/build_mod", methods=['POST'])
def build_mod():
    
    global TRAINED_MODEL
    
    file_obj = request.files.get('traindata')
    print("Type of the file is :", type(file_obj))
    name = file_obj.filename
    print(name)
    file_obj.save(os.path.join(application.config['UPLOAD_FOLDER'],name))
    
    # Is the File extension .csv
    if name.lower().endswith('.csv'):
        print('Input File extension good', name)
    else:
        print('***ERROR*** Input file extension NOT good')
        return render_template('trainform.html', errstr = "***ERROR*** Input file extension NOT good") 
    #End If
    
    # Steps to TRAIN the model
    churn_df = read_data(name)
    disp_df_info(churn_df)
    clean_df = data_cleaning(churn_df)
    x,y=data_split(clean_df)
#   x = feature_engineering(x)
    x = feature_selection(x)
    x = feature_scaling(x)
    TRAINED_MODEL,score = build_logreg_model(x,y)

    return render_template('trainresults.html',acc=score)
# end of home


# In[18]:


# Use model on trainresults.html
# OR Use model on predresults.html
@application.route("/use", methods=['POST','GET'])
def use():
    return render_template('predform.html')
# end of home


# In[19]:


# submit on predform.html
@application.route("/make_pred", methods=['POST'])
def make_pred():
    
    file_obj = request.files.get('newdata')
    print("Type of the file is :", type(file_obj))
    name = file_obj.filename
    print(name)
    file_obj.save(os.path.join(application.config['UPLOAD_FOLDER'],name))
    
    # Is the File extension .csv
    if name.lower().endswith('.csv'):
        print('Input File extension good', name)
    else:
        print('***ERROR*** Input file extension NOT good')
        return render_template('predform.html', errstr = "***ERROR*** Input file extension NOT good") 
    #End If
    
    # Steps to USE model:
    # Call fx Read_data
    new_df = read_data(name)

    # Call fx data_cleaning 
    clean_x = data_cleaning(new_df) 
    print('New Cleaned Data:')
    display(clean_x.head())
    
    # NO Feature Engineering just copy
    new_x = clean_x.copy(deep=True)

    #Feature Selection - Reuse TO_DROP
    #Drop the redundant features
    new_x.drop(new_x[TO_DROP], axis=1, inplace=True)
    print('New Selected Data:')
    display(new_x.head())

    # Feature Scale - Reuse SCALER, TO_SCALE
    dftoscale = new_x[TO_SCALE]

    # Call scaler to scale the df
    dftoscale = SCALER.transform(dftoscale)

    # put the scaled back into original
    new_x[TO_SCALE]=dftoscale
    print('New Scaled Data:')
    display(new_x.head())

    # Make Prediction - Reuse MODEL to make prediction
    new_pred = TRAINED_MODEL.predict(new_x)
    print('New Prediction:',new_pred)

    # new_pred is a np array in a row, transpose to column in order to join with original data frame
    new_pred = np.transpose(new_pred)

    # Add a new column to original data frame called 'Prediction' 
    # with the transposed new_pred np array
    new_df['Prediction']=new_pred


    # Save results to file on server without index
    new_df.to_csv(os.path.join(application.config['UPLOAD_FOLDER'],'result_'+ name),index=False)

    print("*************************** New Prediction Complete WITH FLASK ***************************************")

    # Return results to browser/client, render_template OR send_file , http does NOT allow both.
    # return render_template('predresults.html',data=new_df)
    return(send_file(os.path.join(application.config['UPLOAD_FOLDER'],'result_'+ name),as_attachment=True))

# end of make_pred


# #### Main Program for Web App

# In[20]:


# Main Program for Web app
# If __name__ = __main__ ,program is running standalone
if __name__ == "__main__":
    print("Python script is run standalone")
    print("Python special variable __name__ =", __name__)   
        
       
    # Run the flask app in jupyter noetbook needs run_simple 
    # Run the flask app in python script needs app.run
    # run_simple('localhost',5000, app, use_debugger=True)
    application.run('0.0.0.0',debug=True)

     
else:
    # __name__ will have the name of the module that imported this script
    print("Python script was imported")
    print("Python special variable __name__ =", __name__)   
#End Main program

