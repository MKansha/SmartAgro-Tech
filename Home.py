# ------------ IMPORT LIBRARIES -----------------

import streamlit as st

# st.markdown(f'<h1 style="color:#000000;font-size:36px;">{"SmartAgro Tech"}</h1>', unsafe_allow_html=True)
st.markdown('<h1 style="color: #FFFFFF; font-size: 60px; text-align: center; text-shadow: 2px 2px 4px #000000;">SmartAgro Tech</h1>', unsafe_allow_html=True)



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


add_bg_from_local('Agri.jpg') 

st.markdown(f'<h1 style="color:#FFFFFF;font-size:17px;">{"Agriculture is the one amongst the substantial area of interest to society since a large portion of food is produced by them. Agriculture is the most important sector that influences the economy of India. Predicting crop yield based on the environmental, soil, water and crop parameters has been a potential research topic. Agriculture for years but the results are never satisfying due to various factors that affect the crop yield. Deep-learning-based models are broadly used to extract significant crop features for prediction. Though these methods could resolve the yield prediction problem there exist the following inadequacies: Unable to create a direct non-linear or linear mapping between the raw data and crop yield values; and the performance of those models highly relies on the quality of the extracted features. Finally, the agent receives an aggregate score for the actions performed by minimizing the error and maximizing the forecast accuracy. "}</h1>', unsafe_allow_html=True)


reg = st.button("Register")

# if reg:
    

st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" REGISTER PAGE"}</h1>', unsafe_allow_html=True)




UR = st.text_input("Register User Name",key="username1")
pss1 = st.text_input("First Password",key="password1",type="password")
pss2 = st.text_input("Confirm Password",key="password2",type="password")

# temp_user=[]
    
# temp_user.append(UR)

if pss1 == pss2 and len(str(pss1)) > 2:
    import pandas as pd
    
  
    import csv 
    
    # field names 
    fields = ['User', 'Password'] 
    

    
    # st.text(temp_user)
    old_row = [[UR,pss1]]
    
    # writing to csv file 
    with open(UR+'.csv', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(old_row)
        
    with open('Name.csv', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(old_row)
    st.success('Successfully Registered !!!')        
        
        
    st.success('Successfully Registered !!!')
    
    import pandas as pd
    
    def hyperlink(url):
        return f'<a target="_blank" href="{url}">{url}</a>'
    
    dff = pd.DataFrame(columns=['page'])
    dff['page'] = ['Login']
    dff['page'] = dff['page'].apply(hyperlink)
    dff = dff.to_html(escape=False)

    st.write(dff, unsafe_allow_html=True)     
    

    
else:
    
    st.write('Registeration Failed !!!')     

