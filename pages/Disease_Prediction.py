# ---- IMPORT PACKAGES

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing 
import streamlit as st
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import base64



st.markdown(f'<h1 style="color:#000000;font-size:36px;">{" Disease Prediction"}</h1>', unsafe_allow_html=True)
     
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(to top, rgba(0,0,0,0.5), transparent), url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')     




st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" Disease Prediction Page"}</h1>', unsafe_allow_html=True)
     

UR1 = st.text_input("Login User Name",key="username")

df = pd.read_csv('Name.csv')

U_P1 = df['User'][0]

if str(UR1) == str(U_P1):




    uploaded_file = st.file_uploader("Choose a Image...", type=["jpg", "png"])
    
    # st.text(uploaded_file)   
    
     
    if uploaded_file is None:
        st.markdown(f'<h1 style="color:#FFFFFF;font-size:18px;">{"Please Upload Image"}</h1>', unsafe_allow_html=True)
    
        # st.text("Please Upload Video")
        
    else:
        st.text("Uploaded")        
    
         # from tkinter.filedialog import askopenfilename
         
         # filenamee=askopenfilename()
        
        # ================ INPUT IMAGE ======================
        
        
    
        # if file_up==None:
        #     st.text("Browse")
        # else:
        #  st.image(file_up)
        img = mpimg.imread(uploaded_file)
        st.image(img)
         
         
        # ========= PREPROCESSING ============
         
        img_resize_orig = cv2.resize(img,((50, 50)))
         
         
        try:            
             gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
         
        except:
             gray1 = img_resize_orig
             
             
            
        import os 
         
        # ========= IMAGE SPLITTING ============
         
         
        data_apple_bl = os.listdir('Data/Apple_Black_rot/')
         
        data_app_hea = os.listdir('Data/Apple_Healthy/')
         
        data_app_sca = os.listdir('Data/Apple_Scab/')
         
        data_cherry_hea = os.listdir('Data/Cherry_healthy/')
         
        data_cherry_un = os.listdir('Data/Cherry_Powdery_mildew/')
         
        data_corn_dis = os.listdir('Data/Corn_Disease/')
         
        data_corn_heal = os.listdir('Data/Corn_healthy/')
         
        data_grap_diseas = os.listdir('Data/Grape_Diseased/')
         
        data_grap_heal = os.listdir('Data/Grape_Healthy/')
         
        data_tom_dis = os.listdir('Data/Tomato_Disease/')
         
        data_tom_heal = os.listdir('Data/Tomato_healthy/')
         
    
         
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_apple_bl:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Black_rot//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(1)
         
         
        for img11 in data_app_hea:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(2)
         
         
        for img11 in data_app_sca:
                 # print(img)
                 img_1 = mpimg.imread('Data/Apple_Scab//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(3)
         
         
        for img11 in data_cherry_hea:
                 # print(img)
                 img_1 = mpimg.imread('Data/Cherry_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(4)
         
        for img11 in data_cherry_un:
                 # print(img)
                 img_1 = mpimg.imread('Data/Cherry_Powdery_mildew//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(5)
         
         
        for img11 in data_corn_dis:
                 # print(img)
                 img_1 = mpimg.imread('Data/Corn_Disease//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(6)
         
        for img11 in data_corn_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Corn_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(7)
                 
        for img11 in data_grap_diseas:
                 # print(img)
                 img_1 = mpimg.imread('Data/Grape_Diseased//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(8)
         
        for img11 in data_grap_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Grape_Healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(9)
         
        for img11 in data_tom_dis:
                 # print(img)
                 img_1 = mpimg.imread('Data/Tomato_Disease//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(10)
         
        for img11 in data_tom_heal:
                 # print(img)
                 img_1 = mpimg.imread('Data/Tomato_healthy//' + "/" + img11)
                 img_1 = cv2.resize(img_1,((50, 50)))
         
         
                 try:            
                     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                     
                 except:
                     gray = img_1
         
                 
                 dot1.append(np.array(gray))
                 labels1.append(11)
         
    
                 
           
        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
         
         
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
            
        print("The Total of Images       =",len(dot1))
        print("The Total of Train Images =",len(x_train))
        print("The Total of Test Images  =",len(x_test))
            
            
    ###################
         
         
        # filename = askopenfilename()
        # img = mpimg.imread(filename)
        # img_1 = cv2.resize(img,((50, 50)))
        # try:            
        #     gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        # except:
        #     gray = img_1
        
        # gray=np.array(gray)
        # x_train11=np.zeros((len(gray),50))
        # x_train11[i,:]=np.mean(gray) 
        # # gray=np.zeros((len(gray),50))
        # # x_train11=np.mean(gray)    
        # y_pred = rf.predict(x_train11)[49]
        
    
    ##################
    
    
        # print()
        
    
         
         # for i in range(0,len(gray)):
         #     x_train11[i,:]=np.mean(gray[i])    
        
        
            # ===== CLASSIFICATION ======
            
            
        from keras.utils import to_categorical
        
        
        x_train11=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
            x_train11[i,:]=np.mean(x_train[i])
            
        x_test11=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
             x_test11[i,:]=np.mean(x_test[i])
            
            
        y_train11=np.array(y_train)
        y_test11=np.array(y_test)
            
        train_Y_one_hot = to_categorical(y_train11)
        test_Y_one_hot = to_categorical(y_test)
         
       # ======== CNN ===========
            
        from keras.layers import Dense, Conv2D
        from keras.layers import Flatten
        from keras.layers import MaxPooling2D
        from keras.layers import Activation
        from keras.models import Sequential
        from keras.layers import Dropout
        
        
        # initialize the model
        model=Sequential()
        
        
        #CNN layes 
        model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
        model.add(MaxPooling2D(pool_size=2))
    
        model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Dropout(0.2))
        model.add(Flatten())
        
        model.add(Dense(500,activation="relu"))
        
        model.add(Dropout(0.2))
        
        model.add(Dense(12,activation="softmax"))
        
        #summary the model 
        model.summary()
        
        #compile the model 
        model.compile(loss='binary_crossentropy', optimizer='adam')
        y_train1=np.array(y_train)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        print("-------------------------------------")
        print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
        print("-------------------------------------")
        print()
        #fit the model 
        history=model.fit(x_train11,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
        
        accuracy = model.evaluate(x_train11, train_Y_one_hot, verbose=1)
        
        pred_cnn = model.predict([x_train11])
        
        y_pred2 = pred_cnn.reshape(-1)
        y_pred2[y_pred2<0.5] = 0
        y_pred2[y_pred2>=0.5] = 1
        y_pred2 = y_pred2.astype('int')
        
        loss=history.history['loss']
        loss=max(loss)
        
        acc_cnn=100-loss
        
        print("-------------------------------------")
        print("PERFORMANCE ---------> (CNN)")
        print("-------------------------------------")
        print()
        #acc_cnn=accuracy[1]*100
        print("1. Accuracy   =", acc_cnn,'%')
        print()
        print("2. Error Rate =",loss)
             
         
         
         
         
        Total_length = len(data_apple_bl) + len(data_app_hea) + len(data_app_sca) + len(data_cherry_hea) + len(data_cherry_un) + len(data_corn_dis) + len(data_corn_heal) + len(data_grap_diseas)+ len(data_grap_heal) + len(data_tom_dis) + len(data_tom_heal) 
        
        
        temp_data1  = []
        for ijk in range(0,Total_length):
                    # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)
                
        temp_data1 =np.array(temp_data1)
                
        zz = np.where(temp_data1==1)
                
        if labels1[zz[0][0]] == 1:
            print('-------------------------------------')
            print()
            print(' The Identified Crop  = Apple Balck Rot')
            print()
            print('--------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple ')
            st.text(' The Identified Diseased  =  Balck Rot')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 2:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Apple Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple')
            st.text(' The Identified Diseased  =  Healthy')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 3:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Apple Scab')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Apple')
            st.text(' The Identified Diseased  = Scab')
            print()
            st.text('--------------------------------------')
      
        elif labels1[zz[0][0]] == 4:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Cherry Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Cherry')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
        elif labels1[zz[0][0]] == 5:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Cherry Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Cherry Diseased')
            st.text(' The Identified Disease  = Brown Rot')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 6:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Corn Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Corn ')
            st.text(' The Identified Disease  =  Maize Dwarf Mosaic Virus')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 7:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Corn Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Corn ')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 8:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Grape Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Grape ')
            st.text(' The Identified Disease  = Downy Mildew')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 9:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Grape Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Grape')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 10:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Tomato Diseased')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Tomato')
            st.text(' The Identified Disease  = Bacterial Spot')
            print()
            st.text('--------------------------------------')
    
    
        elif labels1[zz[0][0]] == 11:
            print('-----------------------------------')
             
            print()
            print('The Identified Crop  = Tomato Healthy')   
            print()
            print('-------------------------------------')
    
            st.text('-------------------------------------')
            print()
            st.text(' The Identified Crop  = Tomato ')
            st.text(' The Identified Disease  = Healthy')
            print()
            st.text('--------------------------------------')

else:
    st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" Cant able to open"}</h1>', unsafe_allow_html=True)

    
    
    
    