    
# ------------ IMPORT LIBRARIES -----------------
# ------------ IMPORT LIBRARIES -----------------
import streamlit as st
import pandas as pd
import base64

# Function to add background image
st.markdown(f'<h1 style="color:#000000;font-size:36px;">{" Fertilizer Recommendation"}</h1>', unsafe_allow_html=True)
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

# Function to read fertilizer CSV
def read_fertilizer_csv():
    return pd.read_csv('fertilizer.csv')

# Function to display fertilizer recommendation
def display_fertilizer_recommendation(crop):
    df = read_fertilizer_csv()
    crop_data = df[df['Crop'] == crop]
    if not crop_data.empty:
        st.write("Fertilizer Recommendation for", crop)
        st.write(crop_data.iloc[:, 1:])

        # Add background image for the tabular section
        add_bg_from_local('2.jpg')
        
        # Display types of fertilizers in a tabular format
        st.write("Types of Fertilizers:")
        fertilizers_data = {
            "Type": ["Nitrogen", "Phosphorus", "Potassium", "Compound", "Organic", "Micronutrient",
                     "Liquid", "Slow-release", "Controlled-release"],
            "Examples": [
                "Urea, Ammonium nitrate, Ammonium sulfate, Calcium ammonium nitrate",
                "Superphosphate, Triple superphosphate, Diammonium phosphate",
                "Potassium chloride, Potassium sulfate, Potassium nitrate",
                "10-10-10 (NPK ratio), 20-10-10 (NPK ratio), Custom blends of NPK ratios",
                "Compost, Manure, Bone meal, Fish emulsion, Seaweed extracts",
                "Iron, Zinc, Manganese, Copper, Boron, Molybdenum",
                "Various NPK formulations in liquid form, Foliar fertilizers, Fertigation solutions",
                "Polymer-coated urea, Sulfur-coated urea, Organic slow-release formulations",
                "Coated granules with controlled release mechanisms, Environmentally triggered release formulations"
            ]
        }
        fertilizers_df = pd.DataFrame(fertilizers_data)
        st.table(fertilizers_df)

# UI
add_bg_from_local('2.jpg')
crop = st.selectbox("Select Crop", ["","rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
                                     "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
                                     "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
                                     "coconut", "cotton", "jute", "coffee"])
display_fertilizer_recommendation(crop)

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



st.markdown(f'<h1 style="color:#000000;font-size:36px;">{" Crop Utility"}</h1>', unsafe_allow_html=True)
     
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
add_bg_from_local('2.jpg')     


UR1 = st.text_input("Login User Name",key="username")

df = pd.read_csv('Name.csv')

U_P1 = df['User'][0]

if str(UR1) == str(U_P1):

    st.markdown(f'<h1 style="color:#000000;font-size:16px;">{"Yield advantages:"}</h1>', unsafe_allow_html=True)
         
    import pandas as pd
    
    df = pd.read_csv('Crop.csv')
    
    resultt = df['Crop'][0]
        
        
        
    if resultt == "rice":
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Nitrogen is essential for the vegetative growth of rice plants. * Apply nitrogen in split doses, with the majority being applied at the early vegetative stage and the remaining doses during the active tillering stage. * Nitrogen fertilizer can be applied in the form of urea, ammonium sulfate, or other nitrogen-containing fertilizers."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Phosphorus is crucial for root development, flowering, and seed formation. * Apply phosphorus before transplanting or at the time of transplanting. * Phosphorus-containing fertilizers like single superphosphate or diammonium phosphate can be used."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" *  Potassium is important for overall plant health, disease resistance, and grain filling. * Apply potassium during active tillering and panicle initiation stages. * Potassium-containing fertilizers include potassium chloride, potassium sulfate, or muriate of potash. "}</h1>', unsafe_allow_html=True)
    
    
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)        
    
    
    
    elif resultt == "maize" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Maize, also known as corn, is a versatile grain that offers a variety of health benefits. It is a good source of fiber, vitamins, and minerals."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Maize is a good source of lutein and zeaxanthin, two carotenoids that are important for eye health."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Maize is a good source of fiber, which is important for digestive health."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)    
    
    elif resultt == "chickpea" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Chickpeas are a good source of soluble fiber, which can help to lower LDL (bad) cholesterol levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Chickpeas are low in fat and calories, making them a healthy addition to any diet."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Chickpeas are a good source of iron, folate, manganese, phosphorus, and zinc."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "kidneybeans" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Kidney beans are a good source of fiber, which can help you feel full and satisfied after eating."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Kidney beans are low in saturated fat and cholesterol, and they are a good source of fiber, potassium, and magnesium."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Kidney beans are a good source of fiber, which helps to promote regularity and prevent constipation."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "pigeonpeas" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Pigeon peas are a good source of folate, which is important for the formation of red blood cells."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Pigeon peas are a good source of B vitamins, which help your body convert food into energy."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Pigeon peas are a good source of potassium, which helps to lower blood pressure."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "mothbeans" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Moth beans are a good source of protein and fiber, which can help you feel full and satisfied after eating."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Moth beans have a low glycemic index, which means they do not cause a rapid rise in blood sugar levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Moth beans are a good source of fiber, which is important for digestive health."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "mungbean" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mung beans are a good source of plant-based protein, with one cup of cooked mung beans providing about 14 grams of protein."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mung beans are a good source of both soluble and insoluble fiber. "}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mung beans are a good source of vitamins B6, C, and K. Vitamin B6 is important for brain health and energy production."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "blackgram" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Black gram is a good source of soluble fiber, which can help to lower cholesterol levels and reduce the risk of heart disease."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Black gram is a good source of calcium, magnesium, and phosphorus, which are all important for maintaining bone health."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Black gram has a low glycemic index, which means that it does not cause a rapid spike in blood sugar levels after eating."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "lentil" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Lentils are a good source of fiber, which can help lower LDL (bad) cholesterol levels and blood pressure."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" *  Lentils are a good source of insoluble fiber, which can help promote regular bowel movements."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Lentils contain antioxidants and other compounds that may help protect against certain types of cancer, such as colon cancer."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "pomegranate" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Pomegranates may help lower blood pressure, reduce LDL (bad) cholesterol, and increase HDL (good) cholesterol."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Some studies suggest that pomegranate juice may improve exercise performance and recovery."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Pomegranates may help improve memory and cognitive function."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "banana" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Bananas have a low glycemic index (GI), which means they are unlikely to cause a spike in blood sugar levels after eating."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Bananas are a good source of potassium, which helps to regulate blood pressure."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Bananas are a good source of citrate, which can help to prevent the formation of kidney stones."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "mango" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mangoes are a good source of polyphenols, which are plant compounds with antioxidant properties."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mangoes contain nutrients that support a healthy heart, such as magnesium and potassium, which help maintain healthy blood pressure levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Mangoes are a good source of vitamin C, which is important for immune function."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "grapes" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * rapes are a good source of potassium, which helps balance fluids in the body and lower blood pressure."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Grapes are a rich source of antioxidants, which are compounds that help protect cells from damage caused by free radicals. "}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Grapes contain fiber, which helps slow down the absorption of sugar into the bloodstream. "}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "watermelon" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Watermelon is about 92% water, making it an excellent way to stay hydrated on a hot day. "}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Watermelon contains lycopene, an antioxidant that may help protect against heart disease."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Vitamin C, found in watermelon, is an important antioxidant that helps protect skin cells from damage. "}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "muskmelon" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Muskmelon is about 90% water, making it an excellent source of hydration, especially during hot and humid weather."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Muskmelon is rich in vitamin C, a powerful antioxidant that strengthens the immune system."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Muskmelons vitamin C content contributes to collagen production, which is essential for maintaining skin elasticity and preventing wrinkles."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "apple" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Apples can help lower LDL (bad) cholesterol levels and raise HDL (good) cholesterol levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Apples contain antioxidants, which can help protect cells from damage that can lead to cancer."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Apples can help regulate blood sugar levels, which is important for people with diabetes."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "orange" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Oranges are an excellent source of vitamin C, providing over 100% of the recommended daily intake (RDI) in a medium-sized orange."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Oranges contain compounds that may help protect against heart disease."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * The fiber in oranges helps regulate bowel movements, prevent constipation, and promote a healthy gut microbiome."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "papaya" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Papaya is a good source of fiber, which can help lower cholesterol levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Papaya is a good source of vitamin C, which is an important antioxidant that helps boost the immune system."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Papaya contains several compounds that can help reduce inflammation, such as papain, vitamin C, and vitamin E."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "coconut" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coconut meat is a good source of fiber, which can help promote digestive health and regulate blood sugar levels."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coconut oil has moisturizing and emollient properties that can help hydrate and soften skin."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coconut oil contains antioxidants that may help protect cells from damage caused by free radicals and reduce the risk of chronic diseases."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "jute" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Jute is a renewable resource that can be broken down naturally in the environment. "}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Jute production has a relatively low carbon footprint compared to other fibers."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Jute leaves are a good source of vitamins A, C, and K, as well as minerals such as calcium, iron, and magnesium. "}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
    elif resultt == "coffee" :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coffee consumption may help to protect against liver diseases such as cirrhosis and liver cancer."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coffee may boost metabolism and increase energy expenditure, which may aid in weight control."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Coffee consumption has been linked to a reduced risk of developing certain types of cancer, such as colorectal cancer, liver cancer, and endometrial cancer."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)   
    
       
    else :    
        
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Cotton is a nitrogen-demanding crop. Adequate nitrogen is crucial for vegetative growth and yield. Split the nitrogen application into several doses throughout the growing season to match the crops demand. Early applications are often used to promote early-season growth, and additional applications are made during squaring and boll development stages."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Phosphorus is essential for early root development, flowering, and boll setting. Apply phosphorus based on soil test results and consider using P sources with good availability for the growing season. Incorporate phosphorus into the soil before planting or use a starter fertilizer for improved early-season uptake."}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" * Potassium is critical for boll development, fiber quality, and overall plant health. Cotton removes a significant amount of potassium, so its important to ensure an adequate supply. Apply potassium based on soil test recommendations, and consider using potassium fertilizers with a slow-release component."}</h1>', unsafe_allow_html=True)
        
        
        
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['Disease_Prediction']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)        
    
    
else:
    st.markdown(f'<h1 style="color:#000000;font-size:16px;">{" Cant able to open"}</h1>', unsafe_allow_html=True)