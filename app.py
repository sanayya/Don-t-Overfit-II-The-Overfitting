import pandas as pd
import numpy as np
import prediction
import prediction
import preprocessing
import streamlit as st


st.set_page_config(
     page_title="Don't Overfit II by Sanayya",
     page_icon=None,
     layout="centered",
     initial_sidebar_state="auto",
     menu_items=None
 )


st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Don't Overfit II")

# Uploading CSV file
input_file = st.file_uploader("Upload a CSV File",type=['csv'])

# Creating a Submit button
if st.button("Submit"):

  if (input_file is not None) and input_file.name.endswith(".csv"):
      df = pd.read_csv(input_file)

      # Extracting the id column from the data frame
      df_id_extracted = df[['id']] 
      #SOURCE -->> https://stackoverflow.com/questions/54343898/extract-single-column-from-pandas-dataframe-in-two-ways-difference
      
      # Dropping the id column from the data frame
      df = df.drop("id",axis=1)

      # Pre-processing the data-frame
      preprocessed_df = preprocessing.preprocessing(df)

      # Getting prediction from the saved model
      preds = prediction.final_fun_1(preprocessed_df)
      
      # Creating an empty data frame
      new_df = pd.DataFrame()
      
      # Converting numpy array to data frame
      # preds = pd.DataFrame(preds) 
      # Without the above conversion from Numpy arrays to dataframe, got the error- "TypeError: cannot concatenate object of type '<class 'numpy.ndarray'>'; only Series and DataFrame objs are valid"--->> https://www.marsja.se/how-to-convert-numpy-array-to-pandas-dataframe-examples/

      # Filling the empty df with the predicted values
      new_df["Predicted"] = preds
      

      # Concatenating the two data frames (id + new_df)
      df_concat = pd.concat([df_id_extracted, new_df], axis=1)
      # SOURCE -->> https://datacarpentry.org/python-ecology-lesson/05-merging-data/#:~:text=When%20we%20concatenate%20DataFrames%2C%20we,RIGHT%20of%20the%20first%20DataFrame.
      
      # Displaying the final predicted dataframe
      st.dataframe(df_concat)

    







