import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 


from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_discount_authentication(list1):
    
    
    COLUMN_NAMES = ['STORE_LOCATION', 'PRODUCT_CATEGORY', 'MRP', 'CP', 'SP']
    df1 = pd.DataFrame(columns=COLUMN_NAMES) # Note that there are now row data inserted.
    COLUMN_NAMES_dum = ['MRP', 'CP', 'SP', 'STORE_LOCATION_Denver',
       'STORE_LOCATION_Houston', 'STORE_LOCATION_Miami',
       'STORE_LOCATION_New York', 'STORE_LOCATION_Washington',
       'PRODUCT_CATEGORY_Cosmetics', 'PRODUCT_CATEGORY_Education',
       'PRODUCT_CATEGORY_Electronics', 'PRODUCT_CATEGORY_Fashion',
       'PRODUCT_CATEGORY_Furniture', 'PRODUCT_CATEGORY_Groceries',
       'PRODUCT_CATEGORY_Kitchen']
    
    dum_df = pd.DataFrame(columns=COLUMN_NAMES_dum)



    df2 = pd.DataFrame([list1], columns=['STORE_LOCATION', 'PRODUCT_CATEGORY', 'MRP', 'CP', 'SP'])
    pd.concat([df2, df1])
    dum_df1 = pd.get_dummies(df2)
    df1 = pd.DataFrame(columns=dum_df.columns)
    import numpy as np
    a = np.intersect1d(dum_df.columns, dum_df1.columns)
    frames=[df1,dum_df1]
    df_merge=pd.concat(frames,join='outer', ignore_index=True)
    df_final=df_merge.fillna(0)
    array=classifier.predict(df_final)
    result=array[0]
   
    prediction=classifier.predict(df_final)
    print(prediction)
    return result
