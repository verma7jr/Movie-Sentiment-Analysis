## My Code

# # Step 1: Import Libraries and Load the Model
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model


# # Load the IMDB dataset word index
# word_index = imdb.get_word_index()
# reverse_word_index = {value: key for key, value in word_index.items()}


# # Load the pre-trained model with ReLU activation
# model = load_model('simple_rnn_imdb.h5')


# # Step 2: Helper Functions
# # Function to decode reviews
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# #user input
# user_input=st.text_area("Movie Review")

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')



## Best UI design of above Code
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model("simple_rnn_imdb.h5")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FUNCTIONS ----------------
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    return sequence.pad_sequences([encoded], maxlen=500)

def predict_sentiment(text):
    processed = preprocess_text(text)
    score = model.predict(processed)[0][0]
    label = "Positive" if score > 0.5 else "Negative"
    return label, float(score)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧠 Sentiment AI")
    st.markdown("### Navigation")
    st.markdown("- 🏠 Home")
    st.markdown("- 📊 Analytics (coming soon)")
    st.markdown("- ⚙️ Settings (coming soon)")
    st.divider()
    st.info("RNN-based IMDB classifier")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#111827;'>
        🎬 Movie Review Sentiment Analyzer
    </h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Production-ready AI powered sentiment classification
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- INPUT CARD ----------------
with col1:
    st.markdown("### ✍️ Enter Review")

    user_input = st.text_area(
        "",
        height=180,
        placeholder="Type your movie review here..."
    )

    analyze = st.button("🚀 Analyze", use_container_width=True)

# ---------------- OUTPUT CARD ----------------
with col2:
    st.markdown("### 📊 Result Panel")

    if analyze:
        if user_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            with st.spinner("Running AI model..."):
                label, score = predict_sentiment(user_input)

            # store history
            st.session_state.history.append((user_input, label, score))

            # badge color
            color = "#16a34a" if label == "Positive" else "#dc2626"

            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:12px;
                    background-color:#f9fafb;
                    border:1px solid #e5e7eb;
                    text-align:center;
                ">
                    <h2 style="color:{color}; margin-bottom:5px;">
                        {label}
                    </h2>
                    <p style="color:gray;">Confidence Score</p>
                    <h3>{score:.4f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(score)

# ---------------- HISTORY SECTION ----------------
st.markdown("---")
st.markdown("## 🧾 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions yet.")
else:
    for i, (text, label, score) in enumerate(reversed(st.session_state.history[-5:])):
        st.markdown(
            f"""
            <div style="
                padding:12px;
                margin-bottom:10px;
                border-radius:10px;
                border:1px solid #e5e7eb;
                background-color:#ffffff;
            ">
                <b>{label}</b> | Score: {score:.3f}
                <br>
                <span style="color:gray;">{text[:120]}...</span>
            </div>
            """,
            unsafe_allow_html=True
        )