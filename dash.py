import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Dummy dataset
data = {
    "Text": [
        "AK-47 confiscated in a raid.",
        "No weapons found in the area.",
        "Explosives discovered in the suspect's car.",
        "Pistol recovered from a backpack."
    ],
    "Label": ["Weapon-Related", "Not Weapon-Related", "Weapon-Related", "Weapon-Related"],
}

df = pd.DataFrame(data)

# Dummy function to classify text
def classify_text(text):
    if any(keyword in text.lower() for keyword in ["ak-47", "pistol", "explosives", "weapon"]):
        return "Weapon-Related"
    return "Not Weapon-Related"

# Streamlit dashboard layout
st.title("OSINT Weapon-Related Dashboard")

# Section 1: Display Data
st.header("Dataset Preview")
st.write("Here is a preview of the dummy dataset used:")
st.dataframe(df)

# Section 2: Text Classification
st.header("Text Classification")
input_text = st.text_area("Enter a text to classify:", "Type something about weapons...")
if st.button("Classify"):
    result = classify_text(input_text)
    st.write(f"Classification Result: **{result}**")

# Section 3: Word Cloud Visualization
st.header("Word Cloud for Weapon-Related Texts")
weapon_texts = " ".join(df[df["Label"] == "Weapon-Related"]["Text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(weapon_texts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Section 4: Upload and Analyze Data
st.header("Upload and Analyze")
uploaded_file = st.file_uploader("Upload your CSV file for analysis", type="csv")
if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(uploaded_df.head())

    # Generate Word Cloud for uploaded data
    if "Text" in uploaded_df.columns:
        uploaded_text = " ".join(uploaded_df["Text"].dropna())
        wordcloud_uploaded = WordCloud(width=800, height=400, background_color="white").generate(uploaded_text)

        st.write("Word Cloud for Uploaded Text Data:")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_uploaded, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# Section 5: Visualize Label Distribution
st.header("Label Distribution")
label_counts = df["Label"].value_counts()
st.bar_chart(label_counts)
