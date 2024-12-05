from streamlit_option_menu import option_menu
import streamlit as st
from transformers import pipeline
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import streamlit.components.v1 as components
 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update as per your system
 
def noise_removal(image):
    
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)
 
def extract_text(image):
    preprocessed_image=noise_removal(image)
    text = pytesseract.image_to_string(preprocessed_image, lang='eng', config='--psm 6')
    return text
 
 
# Set up the Streamlit page configuration
st.set_page_config(page_title="SENTIMENT ANALYSIS", page_icon="✨", layout="wide")
 
# Sidebar menu
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #6a0dad; /* Purple background */
            color: white; /* Text color */
        }
        .sidebar .sidebar-content a {
            color: white; /* Link color */
        }
        .sidebar .sidebar-content a:hover {
            background-color: #8a2be2; /* Hover effect */
        }
        .css-1lcbmhc { 
            background-color: #6a0dad; /* Set background for the whole page */
        }
    </style>
""", unsafe_allow_html=True)
 
# Sidebar menu with icons
with st.sidebar:
    page = option_menu(
        menu_title="Explore",  # Menu title
        options=[
            "Introduction",
            "Sentiment Analysis Application",
            "ocr",
            "Datasets Searched",
            "Methods Used",
            "Meta-Analysis Overview",
            "What is Sentiment Analysis?",
            "Common Approaches",
            "Challenges in Sentiment Analysis",
            "Importance of Context",
            "Integration with Other Applications",
            "Sentiment Analysis Application"
        ],  # Menu options
        icons=[
            "info-circle",      # Introduction
            "window-dock",      # Sentiment Analysis Application
            "info-circle",      #ocr
            "search-heart",         # Databasets Searched
            "card-checklist",             # Methods Used
            "backpack3",        # Meta-Analysis Overview
            "question-circle",   # What is Sentiment Analysis?
            "person-raised-hand",   # Common Approaches
            "exclamation-triangle",  # Challenges in Sentiment Analysis
            "book",             # Importance of Context
            "plug",             # Integration with Other Applications
            "window-dock"      # Sentiment Analysis Application
        ],
        menu_icon="cast",
        default_index=0,  # Default selected menu item
 
        styles={
    "container": {"padding": "0!important", "background-color": "transparent", "border-radius": "5px"},  # Transparent background
    "icon": {"color": "#A569BD", "font-size": "18px"},  # Purple icons
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "margin": "0px",
        "padding": "10px 20px",
        "border-radius": "8px",
        "color": "#6C3483",  # Darker purple text
        "font-weight": "bold",
        "transition": "background-color 0.3s ease, color 0.3s ease",
    },
    "nav-link-hover": {
        "background-color": "#D2B4DE",  # Light purple on hover
        "color": "#4A235A"  # Darker purple text on hover
    },
    "nav-link-selected": {
        "background-color": "#8E44AD",  # Deep purple for selected item
        "color": "white" , # White text for contrast
       
    },
}
    )
 
 
# Content display based on sidebar selection

if page == "Introduction":  # Simulating your condition
    st.write("# Sentiment Analysis using Natural Language Processing")

    import streamlit as st

    with st.container():
        components.html(
            """
            <div style="display: flex; justify-content: space-between; gap: 15px; margin-bottom: 10px;">
                <!-- Card 1 -->
                <div style="background-color: #f9f9f9; border: 1px solid #ddd; 
                            border-radius: 8px; padding: 10px; 
                            width: 300px; text-align: center;">
                    <img src="https://miro.medium.com/v2/resize:fit:689/1*jHzNpL-KagnaHUSHzPTPkA.jpeg" 
                        alt="Sentiment Analysis Overview" 
                        style="width: 100%; height: auto; border-radius: 8px;">
                    <p style="margin-top: 8px; font-style: italic; color: #555; font-size: 14px;">Sentiment Analysis Overview</p>
                </div>

                <!-- Card 2 -->
                <div style="background-color: #f9f9f9; border: 1px solid #ddd; 
                            border-radius: 8px; padding: 5px; 
                            width: 300px; text-align: center;">
                    <img src="https://media.sproutsocial.com/uploads/2024/10/Sentiment-analysis-Final.svg" 
                        alt="Natural Language Processing" 
                        style="width: 100%; height: auto; border-radius: 8px;">
                    <p style="margin-top: 8px; font-style: italic; color: #555; font-size: 14px;">Natural Language Processing</p>
                </div>

                <!-- Card 3 -->
                <div style="background-color: #f9f9f9; border: 1px solid #ddd; 
                            border-radius: 8px; padding: 5px; 
                            width: 300px; text-align: center;">
                    <img src="https://cdn.prod.website-files.com/62be0ad3c8b4043cfdca461c/636ec6083e9741367fdca371_sentiment-analysis-hero.webp" 
                        alt="Opinion Mining" 
                        style="width: 100%; height: auto; border-radius: 8px;">
                    <p style="margin-top: 8px; font-style: italic; color: #555; font-size: 14px;">Opinion Mining</p>
                </div>
            </div>
            """,
            height=300,  # Adjust height if necessary
        )

    # Text content
    st.write("### Introduction")
    st.write(   
        """
        A growing number of people around the world are using blogs, forums, and social media platforms like Twitter and Facebook to share their opinions globally. Social media has emerged as one of the most powerful communication tools available. Consequently, this generates a vast amount of data, known as big data, which is analyzed through sentiment analysis to efficiently extract insights. Understanding user sentiment has become essential for industries and organizations.

        Sentiment analysis, also referred to as opinion mining, is a method used to identify whether an author’s or user’s perspective on a topic is positive, neutral or negative. Sentiment analysis involves using natural language processing techniques to extract meaningful information from text and assess the writer’s sentiment, which can be positive, negative, or neutral. Its main goal is to determine polarity and categorize opinion-based texts as either positive or negative. However, sentiment classification is not limited to just positive or negative but can also include categories such as agree or disagree, good or bad. It can even be quantified on a 5-point scale, ranging from strongly disagree to strongly agree. 

        For example, using a rating system of stars from one to five in reviews on electronic home appliances, we can find out the polarity of the above-mentioned items. Sentiment analysis can be further classified into opinion and emotion mining. Though the two seem to be the same, they are slightly different.
        """
    )

elif page == "ocr":
    st.write("### OCR Sentiment Analysis Application")
    
    # Instantiate sentiment analysis pipeline
    sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
 
    # Sidebar instructions
    st.sidebar.header("OCR & Sentiment Analysis Instructions")
    st.sidebar.markdown("""
    1. Upload an image file (JPG, JPEG, or PNG)
    2. Extract text from the image
    3. Analyze the sentiment of extracted text
    """)
 
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Extract text
            st.write("### Extracting Text")
            with st.spinner("Processing..."):
                extracted_text = extract_text(image)
            
            if extracted_text.strip():
                st.subheader("Extracted Text")
                st.text(extracted_text)
                
                # Sentiment Analysis
                if st.button("Analyze Sentiment"):
                    with st.spinner("Analyzing sentiment..."):
                        try:
                            result = sentiment_analysis(extracted_text)
                            if result and isinstance(result, list):
                                sentiment = result[0]
                                sentiment_label = sentiment["label"]
                                confidence = sentiment["score"]
                                emoji = "🙂" if sentiment_label == "POS" else "😐" if sentiment_label == "NEU" else "😞"
                                st.write(f"## Sentiment: {sentiment_label} {emoji}")
                                st.write(f"**Confidence Score:** {confidence:.2f}")
                            else:
                                st.error("Unable to analyze sentiment.")
                        except Exception as e:
                            st.error(f"Error during sentiment analysis: {e}")
            else:
                st.warning("No text was detected in the uploaded image.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please try uploading a different image.")
elif page == "Datasets Searched":
        st.write("### Datasets Searched")
        st.write("Sentiment analysis plays a pivotal role in natural language processing, enabling the detection and categorization of opinions, emotions, and attitudes from textual data. Commonly used datasets for training and testing these models include IMDB movie reviews, Twitter datasets, and e-commerce product reviews. These datasets provide critical insights into various contexts, helping to fine-tune models that can accurately assess sentiment across a range of topics and domains. The dataset in focus here comprises a comprehensive collection of approximately 25,000 tweets that mention Dell during the first three quarters of 2022, spanning from January 1st to September 30th. This dataset is particularly valuable as it provides a real-time snapshot of public perception surrounding a major brand over an extended period. Each tweet entry in the dataset is enriched with key metadata: the date, timestamp, username, and tweet ID, offering granular insights into when, where, and from whom each opinion originated. Additionally, the dataset includes critical annotations for sentiment and emotion, providing both qualitative and quantitative perspectives. The sentiment field identifies the overall polarity of each tweet—positive, negative, or neutral—while the sentiment score adds a numerical value, allowing for finer differentiation within these categories. Similarly, the emotion field captures the primary emotion conveyed (such as joy, anger, surprise, or sadness), and the emotion score provides a metric for the intensity of that specific emotion. Together, these fields offer an intricate look at public sentiment, providing both a categorical label and an intensity measure that enrich the data for more nuanced analyses. This dataset holds substantial potential for research and development within sentiment analysis, machine learning, and brand reputation management. By training models on such a detailed and large-scale dataset, developers and analysts can create systems capable of interpreting sentiment with a high degree of accuracy. Additionally, the dataset's comprehensive nature allows researchers to explore sentiment trends over time, detect seasonal shifts in public perception, and understand the factors that influence changes in consumer sentiment and brand loyalty. With the combination of metadata, sentiment, and emotion scores, this dataset serves as an invaluable tool for capturing the complexities of human emotion and sentiment in digital spaces.")
 
elif page == "Methods Used":
    st.write("### Methods Used")
    st.write("Common methods for sentiment analysis include:")
    st.write("- Lexicon-based approaches: Lexicon-based methods rely on predefined dictionaries of words and phrases associated with positive, negative, or neutral sentiments. Each word in the lexicon is assigned a sentiment score that reflects its emotional tone. When analyzing a piece of text, these approaches aggregate the sentiment scores of the words to determine the overall sentiment. While lexicon-based approaches are simple and interpretable, they may lack the flexibility to handle nuanced language use, such as sarcasm, idioms, or context-dependent meanings. However, they are particularly useful for smaller datasets or in cases where computational resources are limited, offering an efficient and rule-based way to perform sentiment classification.")
    st.write("- Machine Learning models: Machine learning models have become increasingly popular for sentiment analysis due to their ability to learn from data and adapt to diverse language patterns. Common techniques include: Naive Bayes: A probabilistic classifier that applies Bayes' theorem to determine the probability of a sentiment based on word frequency. Naive Bayes is straightforward, fast, and performs well on structured, labeled data, making it effective for basic sentiment classification tasks. Support Vector Machines (SVM): SVMs are robust classifiers that aim to find an optimal boundary between sentiment classes. Known for handling high-dimensional data effectively, SVMs are commonly used in text classification problems, including sentiment analysis, due to their capacity to manage complex datasets. Deep Learning Models: Advanced deep learning models, such as Long Short-Term Memory (LSTM) networks and transformer-based models like BERT (Bidirectional Encoder Representations from Transformers), have further elevated sentiment analysis capabilities. LSTMs are specifically designed to handle sequential data, making them well-suited for understanding context in sentences. BERT and other transformer models, on the other hand, leverage massive training datasets and complex attention mechanisms to capture contextual relationships within text, significantly improving sentiment analysis accuracy for nuanced language.")
    st.write("- Hybrid methods: Hybrid approaches combine the strengths of both lexicon-based and machine learning methods to improve accuracy and adaptability. For instance, a lexicon-based approach may initially classify sentiment, while a machine learning model refines the analysis by adjusting for context and identifying complex sentiment patterns. This combination approach can enhance performance, especially in applications requiring a balance of interpretability and precision. By leveraging the straightforward nature of lexicons and the adaptability of machine learning, hybrid methods offer a powerful tool for sentiment analysis, capable of handling a wide array of text data and emotion nuances.")
 
elif page == "Meta-Analysis Overview":
    st.write("### Meta-Analysis Overview")
    st.write("In the field of sentiment analysis, meta-analysis plays a crucial role by systematically aggregating and analyzing results from multiple studies. This process enables researchers to evaluate the effectiveness of different sentiment analysis methods and approaches, identifying strengths, limitations, and trends across a wide body of research. By combining findings from various studies, meta-analysis provides a comprehensive overview of how specific sentiment analysis techniques perform under diverse conditions, offering valuable insights for both academic research and practical applications. Meta-analysis serves multiple purposes in sentiment analysis. First, it helps researchers to synthesize data on the performance of popular sentiment analysis methods—such as lexicon-based, machine learning, and hybrid approaches. By examining a range of studies, meta-analysis allows researchers to determine which methods consistently yield high accuracy and reliability. For example, while lexicon-based approaches may perform well in straightforward sentiment classification, machine learning and deep learning models like BERT or LSTM may prove more effective in complex contexts where language use is nuanced. Furthermore, meta-analysis can highlight the factors that influence the effectiveness of these methods, such as dataset size, language characteristics, domain specificity, and text complexity. By identifying the conditions under which certain approaches excel, meta-analysis guides practitioners in selecting or designing methods that are best suited for specific applications, whether it’s social media sentiment analysis, customer review mining, or brand reputation monitoring. Meta-analysis in sentiment analysis also has the benefit of uncovering gaps in existing research. It allows researchers to spot areas where further studies are needed, such as evaluating model performance across less common languages, handling mixed emotions, or developing sentiment analysis techniques for niche fields. Additionally, it can reveal emerging trends, such as the growing efficacy of transformer models in multilingual sentiment analysis, encouraging further exploration into cutting-edge methods. Ultimately, meta-analysis offers a robust framework for advancing the field of sentiment analysis. By synthesizing findings across studies, it provides a well-rounded understanding of which methods perform optimally under different scenarios, promoting the development of more accurate and adaptable sentiment analysis models. This, in turn, enables businesses, researchers, and developers to make data-driven decisions and implement sentiment analysis solutions that are best aligned with their unique needs and contexts.")
 
elif page == "What is Sentiment Analysis?":
    st.write("### What is Sentiment Analysis?")
    st.write("Sentiment analysis is a key area in natural language processing (NLP) that focuses on interpreting and extracting subjective information from textual data to gauge the sentiment, emotion, or opinion expressed by the writer. Through the application of advanced NLP techniques, sentiment analysis assesses whether the expressed sentiment is positive, negative, or neutral, providing insights into public opinion, customer satisfaction, and emotional tone. The primary objective of sentiment analysis is to determine the overall polarity of a text, effectively categorizing it based on sentiment and uncovering the writer’s underlying attitude or perspective. In traditional sentiment classification, texts are often categorized simply as positive or negative. However, sentiment analysis can go beyond this binary framework to include additional categories such as ""agree" or "disagree," "good" or "bad," and "other opinion-based classifications. This expanded categorization allows for a more nuanced interpretation of sentiment, capturing subtleties in human expression. For instance, sentiment analysis can recognize varied levels of intensity or certainty in opinions, such as distinguishing between mild agreement and strong endorsement, or between slight criticism and harsh disapproval. To capture even finer distinctions, sentiment analysis can be applied on a quantified scale, such as a 5-point or 7-point rating system. For example, a 5-point scale might range from ""strongly disagree" "to" "strongly agree""", "offering a gradient that quantifies sentiment intensity. This type of scaling is commonly applied in domains like product reviews, where customers rate their experience or satisfaction on a scale from one to five stars. By analyzing these ratings, sentiment analysis can not only classify the review as positive or negative but also assess the degree of satisfaction expressed. For instance, in the context of electronic home appliance reviews, each rating level—from one to five stars—indicates a different level of customer satisfaction, allowing businesses to identify both highly satisfied customers and those who may have concerns. Using a rating-based approach also facilitates the analysis of sentiment trends across a large volume of data. By aggregating ratings from thousands of reviews, sentiment analysis can reveal overall customer sentiment towards a product or service, uncovering valuable insights for quality improvement and customer engagement strategies. In addition, when applied over time, this approach can help track shifts in sentiment, enabling companies to respond proactively to changes in public perception. Overall, sentiment analysis offers a powerful set of tools for understanding and categorizing opinion-based text. By employing binary classifications, expanded sentiment categories, or quantified scales, sentiment analysis provides organizations, researchers, and developers with the flexibility to explore sentiment in depth, making it possible to capture the complexities of human opinion and emotion in a variety of contexts.")
 
elif page == "Common Approaches":
    st.write("### Common Approaches")
    st.write("1. Rule-based methods: Rule-based methods utilize a set of predefined rules and sentiment lexicons to determine the sentiment of a given text. These approaches rely on sentiment dictionaries, which contain words and phrases associated with positive, negative, or neutral sentiments, along with their sentiment scores. Rules are crafted to interpret these words in context, aggregating their sentiment scores to form an overall assessment. For example, rule-based methods may assign higher sentiment weights to words expressing strong emotions, such as ""love" or "hate." "While relatively straightforward and computationally efficient, rule-based approaches often struggle with complex language constructs like sarcasm, irony, and ambiguous terms. Nevertheless, they remain valuable for basic sentiment classification tasks and are commonly used in settings where rapid processing of large datasets is needed with minimal computational resources.")
    st.write("2. Statistical methods: Statistical methods employ various algorithms and statistical models to identify patterns within text data and classify sentiment based on those patterns. Techniques like Naive Bayes, Support Vector Machines (SVM), and logistic regression are commonly used in this approach. Unlike rule-based methods, which depend on predefined lexicons, statistical methods learn from data by identifying word frequency patterns, n-grams, and other features within labeled training data. For instance, in a training dataset of movie reviews, statistical methods can recognize that words like ""excellent" or "terrible" "frequently appear in positive or negative reviews, respectively, and apply this knowledge to classify new, unseen text. Statistical methods are highly effective for basic to moderately complex sentiment classification tasks, and they are flexible enough to handle varied text patterns. However, they often require substantial amounts of labeled data to perform well and may not capture nuanced language as effectively as deep learning models.")
    st.write("3. Deep learning methods: Deep learning methods leverage advanced neural networks to capture complex patterns in text, enabling them to understand sentiment in greater depth and accuracy. These models, such as Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), Convolutional Neural Networks (CNN), and transformer-based models like BERT (Bidirectional Encoder Representations from Transformers), are specifically designed to capture sequential and contextual relationships within text. For instance, LSTMs can learn dependencies across words in a sentence, understanding the flow of sentiment through entire phrases rather than individual words. BERT, on the other hand, uses attention mechanisms to model relationships between words across the entire text, allowing it to handle intricate language nuances, idiomatic expressions, and context-specific meanings. Deep learning methods are highly accurate, especially in complex sentiment analysis tasks involving long texts or multi-layered sentiments, but they typically require large datasets and significant computational resources to train effectively.")
 
elif page == "Challenges in Sentiment Analysis":
    st.write("### Challenges in Sentiment Analysis")
    st.write("Challenges include:")
    st.write("- Ambiguity: Language is inherently ambiguous, and words or phrases can carry different meanings depending on the context in which they appear. For instance, the word “cold” could refer to temperature, an illness, or even a detached personality, depending on the sentence. In sentiment analysis, such ambiguities make it challenging for models to accurately identify sentiment without understanding the broader context. Consider a sentence like “The presentation was cool”—here, “cool” could mean positive approval or a literal temperature, and only the context can clarify which meaning is intended. Addressing ambiguity requires advanced models capable of discerning contextual cues, such as deep learning models that utilize context embeddings. However, even these models struggle with nuanced or heavily contextualized language, making ambiguity a persistent challenge in sentiment analysis.")
    st.write("- Sarcasm: Detecting sarcasm and irony is another significant challenge for sentiment analysis, as these forms of expression often invert the literal meaning of words. In sarcastic statements, positive words can convey negative sentiment, and vice versa, creating difficulties for models trained on direct word-sentiment associations. For example, a tweet saying “Oh great, another Monday!” might appear positive if analyzed based on word choice alone but actually expresses dread or frustration. Sarcasm and irony rely heavily on tone, social cues, and context, elements that are often missing or hard to interpret in text. Overcoming this challenge typically requires advanced sentiment analysis techniques, such as transformer-based models or neural networks trained specifically to recognize patterns indicative of sarcasm. Even so, accurately detecting sarcasm remains complex and can lead to lower accuracy in sentiment analysis results.")
    st.write("- Domain-specific language: Sentiment analysis models often need to be customized to the specific language and terminology of different domains. Words and phrases used in one field may have entirely different meanings in another, and models trained on general datasets may misinterpret sentiment when applied to specialized domains. For instance, in the financial sector, words like “bullish” or “bearish” hold distinct connotations related to market trends, while in healthcare, terms like “positive” or “negative” have diagnostic implications rather than emotional ones. Domain-specific language also appears in industry-specific slang or product jargon, requiring sentiment models to be tailored to understand these unique lexicons. To address this, sentiment analysis models often need retraining or fine-tuning on domain-specific data, which can be both time-consuming and resource-intensive. Additionally, the scarcity of labeled datasets in niche domains can limit the effectiveness of such models, posing a further challenge in adapting sentiment analysis for specialized fields.")
 
elif page == "Importance of Context":
    st.write("### Importance of Context")
    st.write("In sentiment analysis, accurately interpreting the context in which text is written is essential for capturing the true sentiment conveyed. Context shapes how words, phrases, and expressions are understood, directly influencing the sentiment and emotional tone of the text. Sentiment analysis models, therefore, need to be trained to recognize and adapt to these contextual nuances to make reliable assessments. Contextual understanding in sentiment analysis is multifaceted. A word that typically carries a positive or negative connotation may shift in meaning depending on the surrounding words, sentence structure, or even cultural references. For instance, the phrase ""That's just fantastic" "could convey genuine enthusiasm or frustration depending on the context in which it is used. Without this contextual insight, sentiment analysis models may misinterpret the sentiment, mistaking sarcasm for positivity or overlooking subtle tones of discontent. Context becomes even more critical in long or complex texts, where sentiments can evolve and change throughout a passage. For instance, a customer review might start with praise for certain features but shift to criticism later. A sentiment analysis model that doesn’t consider context may overemphasize the positive introduction and fail to capture the negative sentiment in the latter part of the text. Similarly, context can vary significantly across different communication platforms—text messages, emails, and social media posts each have unique norms and language styles that influence the sentiment. Social media posts, for example, may use slang, emojis, or hashtags that require a specific contextual understanding for accurate interpretation. Recognizing contextual nuances also extends to domain-specific language, where words and expressions carry meanings specific to a field or industry. For instance, in the context of financial discussions, the term ""risk" "may be neutral or even positive, associated with potential rewards. In healthcare, however, the same term likely carries a more negative connotation. Sentiment analysis models trained on general datasets may misinterpret these domain-specific sentiments if they lack the contextual knowledge specific to that field. Advanced sentiment analysis models, such as those based on deep learning and transformer architectures, have made significant strides in contextual understanding. These models utilize context embeddings and attention mechanisms, allowing them to weigh different parts of a text and capture nuanced relationships between words. BERT, for example, can analyze both left and right contexts within a sentence, enabling it to interpret words based on their surrounding text. Such models are better equipped to handle contextual nuances, providing a more accurate sentiment analysis by recognizing shifts in tone, sarcasm, and domain-specific meanings.")
 
elif page == "Integration with Other Applications":
    st.write("### Integration with Other Applications")
    st.write("Sentiment analysis is a powerful tool that enhances various applications such as customer feedback systems, social media monitoring tools, and market research platforms, providing organizations with actionable insights from textual data. In customer feedback systems, sentiment analysis transforms raw customer opinions into meaningful insights, allowing businesses to understand satisfaction levels, address concerns promptly, and refine products based on user feedback. Integrated into social media monitoring tools, sentiment analysis enables companies to track public opinion and brand perception across platforms like Twitter and Facebook, quickly detecting sentiment shifts and potential crises to safeguard brand reputation. In market research, sentiment analysis helps organizations understand consumer preferences, identify industry trends, and assess competitors by analyzing sentiment in surveys, reviews, and forums. Through these diverse applications, sentiment analysis enables organizations to harness real-time feedback and sentiment trends, empowering them to enhance customer satisfaction, respond proactively to market shifts, and make data-driven strategic decisions for continuous improvement.")
 
elif page == "Sentiment Analysis Application":
    st.write("# Sentiment Analysis Application")
    st.write("This app uses a Hugging Face model to analyze the sentiment of the text you provide. Type in your text and click 'Analyze' to see whether the sentiment is positive, negative, or neutral.")
 
    # Load pre-trained sentiment analysis model from Hugging Face
    sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
 
    # User Input
    user_input = st.text_area("Enter your text here:")
 
    # Analyze Button
    if st.button("Analyze"):
        if user_input.strip() != "":
            try:
                result = sentiment_analysis(user_input)
                if isinstance(result, list) and len(result) > 0:
                    sentiment = result[0]
                    val = "😞" if sentiment['label'] == "NEG" else "🙂" if sentiment['label'] == "POS" else "😐"
                    st.write(f"""
                    ## Sentiment: {sentiment['label']} {val}
                    Confidence Score: {sentiment['score']:.2f}
                    """)
                else:
                    st.write("Unexpected result format.")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter some text before clicking Analyze.")
 
# Footer
st.write("""
---
### Contact Us
For any questions or support, please reach out to us at:
- Email: support@codezenith.com
- Phone: +8336909143
 
Created with ❤ by CodeZenith
""")