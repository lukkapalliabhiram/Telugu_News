import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import linear_kernel
from indicnlp.tokenize import indic_tokenize
import string
import regex

# Function definitions

def custom_analyzer(text):
    words = regex.findall(r'\w{1,}', text)
    for w in words:
        yield w

def predict_text_sample(test_text, inv_topic_dict, clf, count_vec):
    test_sample = [test_text]
    x_test_sample_features = count_vec.transform(test_sample)
    y_pred_test_sample = clf.predict(x_test_sample_features)
    return inv_topic_dict[y_pred_test_sample[0]]

def classify_and_recommend(article_text, clf, count_vec, df_content):
    article_vector = count_vec.transform([article_text])
    predicted_class = clf.predict(article_vector)[0]
    filtered_articles = df_content[df_content['topic'] == predicted_class].reset_index(drop=True)
    recommendations, similarity_scores = get_recommendations_with_scores(article_text, filtered_articles)
    return predicted_class, recommendations, similarity_scores

def get_recommendations_with_scores(name, df_content):
    vect = TfidfVectorizer(analyzer=custom_analyzer, ngram_range=(1,2), max_df=0.85, min_df=0.05)
    count_matrix = vect.fit_transform(df_content.body_processed.values)
    new_vec = vect.transform([name])
    cosine_sim = linear_kernel(new_vec, count_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: (x[1], df_content['date'][x[0]]), reverse=True)
    sim_scores = sim_scores[0:10]
    article_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    return df_content['body_processed'].iloc[article_indices], scores

# Preprocessing function
def preprocess_data(df):
    df["body_processed"] = df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    df["body_processed"] = df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
    return df

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    train_path = "./train_telugu_news.csv"
    test_path = "./test_telugu_news.csv"
    telugu_news_df = pd.read_csv(train_path)
    test_news_df = pd.read_csv(test_path)

    # Preprocess steps
    # Removing special characters and unnecessary white spaces
    telugu_news_df["body_processed"] = telugu_news_df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    telugu_news_df["body_processed"] = telugu_news_df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    test_news_df["body_processed"] = test_news_df["body"].str.replace('\u200c', '').str.replace('\n', '').str.replace('\t', '').str.replace('\xa0', '')
    test_news_df["body_processed"] = test_news_df["body_processed"].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    # Categorical Encoding of Topics
    topic_dic = {topic: idx for idx, topic in enumerate(telugu_news_df["topic"].unique())}
    inv_topic_dict = {v: k for k, v in topic_dic.items()}
    
    telugu_news_df["topic"] = telugu_news_df["topic"].map(topic_dic)
    test_news_df["topic"] = test_news_df["topic"].map(topic_dic)

    return telugu_news_df, test_news_df, inv_topic_dict

telugu_news_df, test_news_df, inv_topic_dict = load_and_preprocess_data()


# Model training and feature extraction code
# ...
# Feature Extraction - Preparing the training and testing data
categories = [i for i in range(len(telugu_news_df["topic"].unique()))]
text_topic = [' '.join(telugu_news_df[telugu_news_df["topic"] == i]["body_processed"].tolist()) for i in categories]

# Using CountVectorizer for feature extraction
count_vec = CountVectorizer(max_df=0.75, min_df=0.1, lowercase=False, analyzer=custom_analyzer, max_features=100000, ngram_range=(1,2))
x_train_features = count_vec.fit_transform(text_topic)

# Preparing the labels
y_train = categories

# Model Training
clf = MultinomialNB()
clf.fit(x_train_features, y_train)

# Preparing the test data for prediction
x_test = test_news_df["body_processed"].tolist()
y_test = test_news_df["topic"].tolist()
x_test_features = count_vec.transform(x_test)


# Streamlit App
def main():
    st.title("Telugu News Classification and Recommendation")

    # Custom CSS styling
    st.markdown("""
        <style>
        .main {
            background-color: #F5F5F5;
        }
        .stButton > button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
            border: none;
            padding: 10px 24px;
            margin: 10px;
        }
        .stExpander > label {
            font-size: 18px;
            font-weight: bold;
        }
        .article-body {
            white-space: pre-line;
        }
        </style>
    """, unsafe_allow_html=True)

    # User input
    user_input = st.text_area("Enter a Telugu News Article:")

    if st.button("Classify and Recommend"):
        if user_input:
            predicted_class, recommendations, scores = classify_and_recommend(user_input, clf, count_vec, telugu_news_df)
            st.write(f"Predicted Class: {inv_topic_dict[predicted_class]}")

            st.write("Recommended Articles:")
            for rec, score in zip(recommendations, scores):
                matched_articles = telugu_news_df[telugu_news_df['body_processed'] == rec]
                if not matched_articles.empty:
                    heading = matched_articles.iloc[0]['heading']
                    body = rec

                    with st.expander(f"See details for: {heading}"):
                        st.markdown(f"<div class='article-body'><strong>Heading:</strong> {heading}<br><br><strong>Body:</strong> {body}<br><br><strong>Similarity Score:</strong> {score:.4f}</div>", unsafe_allow_html=True)
        else:
            st.write("Please enter an article for classification.")

if __name__ == "__main__":
    main()
