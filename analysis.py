import streamlit as st
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords
import heapq
import re
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.tokenized_sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
    
    def named_entity_recognition(self):
        doc = self.nlp(self.text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def topic_modeling(self, num_topics=5):
        # Concatenate tokenized sentences into strings
        text_concatenated = [' '.join(sentence) for sentence in self.tokenized_sentences]

        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(text_concatenated)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        return lda_model, vectorizer

    def language_detection(self):
        doc = self.nlp(self.text)
        return doc.lang_

    def part_of_speech_tagging(self):
        doc = self.nlp(self.text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return pos_tags

    def dependency_parsing(self):
        doc = self.nlp(self.text)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        return dependencies

    def get_sentiment_score(self, selected_text):
        return self.sentiment_analyzer.polarity_scores(selected_text)['compound']

    def extract_keywords(self, num_keywords=5):
        words = word_tokenize(self.text)
        filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in self.stop_words]
        word_freq = FreqDist(filtered_words)
        return word_freq.most_common(num_keywords)

    def search_text(self, query):
        matches = re.findall(r'\b{}\b'.format(re.escape(query)), self.text, flags=re.IGNORECASE)
        return matches

    def calculate_word_count(self):
        words = word_tokenize(self.text)
        return len(words)

    def calculate_sentence_lengths(self):
        sentences = sent_tokenize(self.text)
        return [len(word_tokenize(sentence)) for sentence in sentences]

    def calculate_paragraph_lengths(self):
        paragraphs = [p.strip() for p in self.text.split('\n\n') if p.strip()]
        return [len(word_tokenize(paragraph)) for paragraph in paragraphs]

    def get_themes(self):
        paragraphs = [p.strip() for p in self.text.split('\n\n') if p.strip()]
        themes = []
        for paragraph in paragraphs:
            words = [word.lower() for word in word_tokenize(paragraph) if word.isalnum() and word.lower() not in self.stop_words]
            freq_dist = FreqDist(words)
            if freq_dist:
                main_idea = max(freq_dist, key=freq_dist.get)
                themes.append(main_idea)
        return themes

    def sentence_summarization(self, num_sentences=5):
        sentences = sent_tokenize(self.text)
        word_freq = FreqDist(word.lower() for word in sentences if word.lower() not in self.stop_words)
        sentence_scores = {}
        for sentence in sentences:
            for word, freq in word_freq.items():
                if word in sentence.lower():
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        return summary_sentences
    
    def generate_word_cloud(text, max_words=150, background_color='white'):
        wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color=background_color).generate(text)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud from Text')
        plt.close(fig)  # Close the figure to prevent it from being displayed immediately
        return fig


# Streamlit app
def main():
    st.title("PDF Word Cloud Generator and Text Analysis")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text(uploaded_file)
        analyzer = TextAnalyzer(pdf_text)

        st.subheader("Named Entity Recognition:")
        entities = analyzer.named_entity_recognition()
        st.write("Entities:", entities)

        st.subheader("Language Detection:")
        language = analyzer.language_detection()
        st.write("Language:", language)

        st.subheader("Part-of-Speech Tagging:")
        pos_tags = analyzer.part_of_speech_tagging()
        st.write("POS Tags:", pos_tags)

        st.subheader("Dependency Parsing:")
        dependencies = analyzer.dependency_parsing()
        st.write("Dependencies:", dependencies)

        st.subheader("Sentiment Analysis Results:")
        st.write("Sentiment Score:", analyzer.sentiment_analyzer.polarity_scores(pdf_text)['compound'])

        selected_text = st.text_area("Enter text for sentiment analysis:", value="")
        if selected_text:
            st.write("Sentiment Score:", analyzer.get_sentiment_score(selected_text))

        metric_filter = st.selectbox("Choose Metric", ["Word Count", "Sentence Length", "Paragraph Length"])
        st.subheader(f"{metric_filter} Metrics:")
        if metric_filter == "Word Count":
            st.write(f"{metric_filter}: {analyzer.calculate_word_count()}")
        elif metric_filter == "Sentence Length":
            st.bar_chart(analyzer.calculate_sentence_lengths())
            st.write(f"{metric_filter} Distribution")
        elif metric_filter == "Paragraph Length":
            st.bar_chart(analyzer.calculate_paragraph_lengths())
            st.write(f"{metric_filter} Distribution")

        st.subheader("Keywords:")
        num_keywords = st.slider("Number of Keywords to Display", min_value=1, max_value=20, value=5)
        keywords = analyzer.extract_keywords(num_keywords)
        sort_option = st.radio("Sort by", ["Occurrence", "Alphabetically"])
        if sort_option == "Occurrence":
            keywords.sort(key=lambda x: x[1], reverse=True)
        else:
            keywords.sort(key=lambda x: x[0])
        st.write("Top Keywords:", keywords)

        st.subheader("Word Cloud:")
        max_words = st.slider("Maximum Words in Word Cloud", min_value=50, max_value=500, value=150, step=50)
        background_color = st.color_picker("Background Color", "#FFFFFF")
        word_cloud = WordCloud(width=800, height=400, max_words=max_words, background_color=background_color).generate(pdf_text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud from PDF Text')
        st.pyplot(fig)
        
        st.subheader("Search Text:")
        search_query = st.text_input("Enter text to search")
        if search_query:
            search_results = analyzer.search_text(search_query)
            st.write("Search Results:", search_results)

        st.subheader("Sentence Analysis:")
        analysis_choice = st.radio("Choose Analysis", ["Themes", "Summarization", "Topic Modeling"])
        if analysis_choice == "Themes":
            themes = analyzer.get_themes()
            st.write("Themes:", themes)
        elif analysis_choice == "Summarization":
            num_summary_sentences = st.slider("Number of Sentences for Summarization", 1, 10, 5)
            summary_sentences = analyzer.sentence_summarization(num_summary_sentences)
            st.write("Summary Sentences:")
            for sentence in summary_sentences:
                st.write("- ", sentence)
        elif analysis_choice == "Topic Modeling":
            num_topics = st.slider("Number of Topics for Topic Modeling", 2, 10, 5)
            lda_model, vectorizer = analyzer.topic_modeling(num_topics)
            for index, topic in enumerate(lda_model.components_):
                st.write(f"Topic {index + 1}:")
                st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

      
if __name__ == "__main__":
    main()