# ------------ IMPORT LIBRARIES -----------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


st.markdown(f'<h1 style="color:#000000;font-size:36px;">{" Crop Recommendation"}</h1>', unsafe_allow_html=True)


import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')     



st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" Crop recommendation Page"}</h1>', unsafe_allow_html=True)

UR1 = st.text_input("Login User Name",key="username")

df = pd.read_csv('Name.csv')

U_P1 = df['User'][0]

if str(UR1) == str(U_P1):
    

    print("---------------------------------------------")
    print(" Input Data ---> Crop Recommendation")
    print("---------------------------------------------")
    print()
    
    
    # ================  INPUT  ===
    
    df=pd.read_csv("Crop_recommendation.csv")    
    # df=df[0:2500]
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print(df.head(15))
    
    
    # ================  PRE-PROCESSING  ===
     
     # --- MISSING VALUES 
     
    print("--------------------------------")
    print("  Handling Missing Values")
    print("--------------------------------")                    
    print(df.isnull().sum())    
    res=df.isnull().sum().any()
     
    if res==False:
        print("---------------------------------------------")
        print("There is no missing values in our dataset !!!")
        print("---------------------------------------------")
    else:
        print("---------------------------------------------")
        print("Missing values is present in our dataset !!!")
        print("---------------------------------------------")  
        
        
    print("----------------------------------------------------")
    print("Before Label Encoding          ")
    print("----------------------------------------------------")
    print()
    print(df['label'].head(15))
    print()
    
    data_label = df['label']
    label_encoder=preprocessing.LabelEncoder()
    
    print("----------------------------------------------------")
    print("After Label Encoding          ")
    print("----------------------------------------------------")
    print()
    
    df['label']=label_encoder.fit_transform(df['label'])
    
    print(df['label'].head(15))
        
    #============================= DATA SPLITTING ==============================
    
    
    print("----------------------------------------------------")
    print("Data Splitting          ")
    print("----------------------------------------------------")
    print()
    
    from sklearn.model_selection import train_test_split
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=100)
    
    print("Total no of data's       :",df.shape[0])
    print()
    print("Total no of Train data's :",X_train.shape[0])
    print()
    print("Total no of Test data's  :",X_test.shape[0])
    
    
    # ============ RANDOM FOREST ===================
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier()
    
    rf.fit(X_train,y_train)
    
    pred_rf = rf.predict(X_test)
    
    from sklearn import metrics
    
    acc_rf = metrics.accuracy_score(y_test, pred_rf) * 100
    
    print("----------------------------------")
    print("ML --> Random Forest Classifier  ")
    print("----------------------------------")
    print()
    print("1) Accuracy = ", acc_rf,'%')
    print()
    print("2) Classification Report ")
    print()
    print(metrics.classification_report(y_test, pred_rf))
    
    
    st.text("----------------------------------")
    st.text("ML --> Random Forest Classifier  ")
    st.text("----------------------------------")
    
    st.write("1) Accuracy = ", acc_rf,'%')
    print()
    print("2) Classification Report ")
    print()
    # st.write(metrics.classification_report(y_test, pred_rf))
    metrics.classification_report(y_test, pred_rf)
    
    
    
    st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Enter the following details"}</h1>', unsafe_allow_html=True)
    
    
    a1 = st.text_input("Enter Nitrogen Value")
    
    a2 = st.text_input("Enter Phosporros")
    
    a3 = st.text_input("Enter Potassium") 
    
    a4 = st.text_input("Enter Temperature")
    
    a5 = st.text_input("Enter Humidity ")
    
    a6 = st.text_input("Enter Ph value") 
    
    a7 = st.text_input("Enter Rainfall") 
    
    aa = st.button('Submit')
    
    if aa:
        Data_reg = [a1,a2,a3,a4,a5,a6,a7]
                    
        y_pred_reg=rf.predict([Data_reg])
        
        pred = label_encoder.inverse_transform(y_pred_reg)
        
        # st.text(pred)
        
        # res = data_label[y_pred_reg]
        
        # res = res.to_string(index=False, header=False)        
        
        st.write("------------------------------")
        st.write("The Identified Crop = ", pred[0])    
        st.write("------------------------------")
    
        # import pickle
        # with open('Result.pickle', 'wb') as f:
        #     pickle.dump(res, f)          
            
        # fert_data = pd.read_csv("fertilizer.csv")
            
        import pandas as pd
        
      
        import csv 
        
        # field names 
        fields = ['Crop'] 
        
    
        
        # st.text(temp_user)
        old_row = [[pred[0]]]
        
        # writing to csv file 
        with open('Crop.csv', 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(old_row)   
        

        
    
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Fertilizer']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)       
    
    
else:
    st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" Cant able to open"}</h1>', unsafe_allow_html=True)

    
    
    
    
    
    
    
    
    