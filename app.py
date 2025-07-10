import streamlit as st
import joblib

#Loads the TF-IDF vectorizer 
#Loads the Logistic Regression model 

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("lr_model.joblib")

#Displays a large title on the web app
st.title("Fake News Detector")

#Adds some instructional text 
st.write("Enter a News Article below to check whether it is Fake or Real. ")

#cteate multiline text box
inputn = st.text_area("News Article:","")

if st.button("Check News"):

    if inputn.strip():
        #Checks whether the user actually entered something
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
    else:
        st.warning("Please enter some text to Analyze. ") 