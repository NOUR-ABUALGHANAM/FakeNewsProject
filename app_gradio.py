import gradio as gr
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLP resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load trained model and vectorizer
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_news(text):
    if text.strip() == "":
        return "⚠️ Please enter a news article."
    clean_text = preprocess_text(text)
    vec = vectorizer.transform([clean_text])
    prediction = model.predict(vec)[0]
    return "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS"

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste a news article here..."
    ),
    outputs="text",
    title="📰 Fake News Detection System",
    description=(
        "This web-based application uses Natural Language Processing (NLP) "
        "and Machine Learning to classify news articles as Fake or Real.\n\n"
        "The deployed model is **Logistic Regression**, trained on a large real-world dataset."
    ),
    examples=[
        ["The World Health Organization announced new vaccination guidelines to improve global health safety."],
        ["Scientists confirmed that drinking cola every morning can cure diabetes in one week."],
        ["The following statementsÂ were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own.Â Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump : - Vanity Fair, which looks like it is on its last legs, is bending over backwards in apologizing for the minor hit they took at Crooked H. Anna Wintour, who was all set to be Amb to Court of St Jamesâ€™s & a big fundraiser for CH, is beside herself in grief & begging for forgiveness! [1024 EST] -- Source link: (bit.ly/2jBh4LU) (bit.ly/2jpEXYR)"]
    ]
)

interface.launch(share=True)
