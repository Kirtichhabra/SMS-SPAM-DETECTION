import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

# Ensure that nltk stopwords are downloaded
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open(r'C:\Users\Testbook\Downloads\sms-spam-prediction2\model.pkl', 'rb'))
vectorizer = pickle.load(open(r'C:\Users\Testbook\Downloads\sms-spam-prediction2\vectorizer.pkl', 'rb'))

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("Spam Email Detection Using NLP and PyCaret")

# Input form
user_input = st.text_area("Enter the email content here", height=200)

# Detect button
if st.button("Detect Spam"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)

        # Vectorize the input text
        input_vec = vectorizer.transform([processed_input])

        # Predict using the loaded model
        prediction = model.predict(input_vec.toarray())

        # Display the result
        if prediction[0] == 1:
            st.error("This is a Spam email.")
        else:
            st.success("This is not a Spam email.")
