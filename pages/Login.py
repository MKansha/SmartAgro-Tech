# ------------ IMPORT LIBRARIES -----------------

import streamlit as st
import pandas as pd


st.markdown(f'<h1 style="color:#FFFFFF;font-size:36px;">{"Crop Harvesting"}</h1>', unsafe_allow_html=True)


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
add_bg_from_local('1.jpg') 
   

st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" LOGIN PAGE"}</h1>', unsafe_allow_html=True)


UR1 = st.text_input("Login User Name",key="username")
psslog = st.text_input("Password",key="password",type="password")
# tokenn=st.text_input("Enter Access Key",key="Access")
agree = st.checkbox('LOGIN')
    
if agree:
    try:
        
        df = pd.read_csv(UR1+'.csv')
        U_P1 = df['User'][0]
        U_P2 = df['Password'][0]
        if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
            st.success('Successfully Login !!!')    
            
            def hyperlink(url):
                    return f'<a target="_blank" href="{url}">{url}</a>'
                
            dff = pd.DataFrame(columns=['page'])
            dff['page'] = ['Crop_Recommendation']
            dff['page'] = dff['page'].apply(hyperlink)
            dff = dff.to_html(escape=False)
            
            st.write(dff, unsafe_allow_html=True)        

            
          
        else:
            st.write('Login Failed!!!')
    except:
        st.write('Login Failed!!!')        