import streamlit as st
from transformers import pipeline

# Configure page
st.set_page_config(page_title="SENTIMENT ANALYSIS", page_icon="📖", layout="wide")

# Navigation Bar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    [
        "Introduction",
        "Databases Searched",
        "Methods Used",
        "Meta-Analysis Overview",
        "What is Sentiment Analysis?",
        "Common Approaches",
        "Challenges in Sentiment Analysis",
        "Importance of Context",
        "Integration with Other Applications",
        "Sentiment Analysis Application"
    ]
)

# Page Title
st.title("Sentiment Analysis in Natural Language Processing")

# Content based on navigation
if section == "Introduction":
    st.header("Introduction")
    st.write("Sentiment analysis, also known as opinion mining, involves determining the emotional tone behind a series of words. This process is used to understand attitudes, opinions, and emotions expressed in text.")

elif section == "Databases Searched":
    st.header("Databases Searched")
    st.write("For sentiment analysis, databases such as IMDB movie reviews, Twitter datasets, and product reviews from e-commerce sites are commonly used. These datasets help in training and testing sentiment analysis models.")

elif section == "Methods Used":
    st.header("Methods Used")
    st.write("Common methods for sentiment analysis include:")
    st.write("- **Lexicon-based approaches**: These use predefined lists of words and phrases to determine sentiment.")
    st.write("- **Machine Learning models**: Techniques such as Naive Bayes, Support Vector Machines, and deep learning models like LSTM and BERT are used.")
    st.write("- **Hybrid methods**: Combining lexicon-based and machine learning approaches to improve accuracy.")

elif section == "Meta-Analysis Overview":
    st.header("Meta-Analysis Overview")
    st.write("Meta-analysis in sentiment analysis involves aggregating results from different studies to assess the effectiveness of various methods and approaches. It provides insights into which methods perform best under different conditions.")

elif section == "What is Sentiment Analysis?":
    st.header("What is Sentiment Analysis?")
    st.write("Sentiment analysis aims to classify text into categories such as positive, negative, or neutral. It helps businesses understand customer opinions and can be used to monitor brand reputation and customer satisfaction.")

elif section == "Common Approaches":
    st.header("Common Approaches")
    st.write("1. **Rule-based methods**: Use predefined rules and lexicons to analyze sentiment.")
    st.write("2. **Statistical methods**: Employ algorithms to identify patterns and classify sentiment.")
    st.write("3. **Deep learning methods**: Use neural networks to capture complex patterns in text.")

elif section == "Challenges in Sentiment Analysis":
    st.header("Challenges in Sentiment Analysis")
    st.write("Challenges include:")
    st.write("- **Ambiguity**: Words can have different meanings depending on context.")
    st.write("- **Sarcasm**: Detecting sarcasm and irony remains difficult.")
    st.write("- **Domain-specific language**: Sentiment analysis models need to be adapted for different domains.")

elif section == "Importance of Context":
    st.header("Importance of Context")
    st.write("Understanding the context in which text is written is crucial for accurate sentiment analysis. Context can influence the sentiment expressed, and models must be trained to recognize contextual nuances.")

elif section == "Integration with Other Applications":
    st.header("Integration with Other Applications")
    st.write("Sentiment analysis can be integrated into various applications such as customer feedback systems, social media monitoring tools, and market research platforms to provide actionable insights.")

elif section == "Sentiment Analysis Application":
    st.header("Sentiment Analysis Application")
    st.write("""
    This app uses a **Hugging Face** model to analyze the sentiment of the text you provide. 
    Type in your text and click "Analyze" to see whether the sentiment is positive, negative, or neutral.
    """)

    # User input for sentiment analysis
    user_input = st.text_area("Enter the text you want to analyze:")

    if st.button("Analyze"):
        st.write("Analyzing...")
        with st.spinner("Analyzing..."):
            # Load pre-trained sentiment analysis model
            sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
            analysis_result = sentiment_analysis(user_input)
        
        label = analysis_result[0]["label"]
        score = analysis_result[0]['score']
        
        st.write("Analysis Result:")
        if label == 'NEG':
            st.warning(f"NEGATIVE with a score of {score}")
        elif label == 'POS':
            st.success(f"POSITIVE with a score of {score}")
        else:
            st.write(f"NEUTRAL with a score of {score}")

# Footer
st.write("""
---
### Contact Us
For any questions or support, please reach out to us at:
- **Email:** support@codezenith.com
- **Phone:** +8336909143

Created with ❤️ by CodeZenith
""")
