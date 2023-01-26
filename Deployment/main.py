import pandas as pd
import numpy as np
import streamlit as st
#import altair
import joblib



pipe_lr = joblib.load(open("Hotel_Review.pkl","rb"))

def predict(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

dict = {0: "Negative", 1: "positive"}

def main():

    st.title("Hotel Review Classification")
    menu= ["Home", "About"] 
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Hotel Review In Text")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label='Predict')
            
            if submit_text:
                col1,col2 = st.columns(2)
                #apply fun here
                prediction = predict(raw_text)
                probability = get_prediction_proba(raw_text)
                
                with col1:
                  st.success("Original Text")
                  st.write(raw_text)
                    
                  st.success("Prediction")
                  icon = dict[prediction]
                  st.write("{}:{}".format(prediction, icon))
                  st.write("Confidence:{}".format(np.max(probability)))
                  
                  with col2:
                   #st.success("Prediction Probability")
                   #st.write(probability)
                   proba_df= pd.DataFrame(probability, columns=pipe_lr.classes_)
                   #st.write(proba_df.T)
                   proba_df_clean =proba_df.T.reset_index()
                   proba_df_clean.columns = ["sentiment", "probability"]
                
        

    else:
        st.subheader("About")
        
        
if __name__ == '__main__':
   main()
        
