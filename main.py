import streamlit as st
from pdfminer.high_level import extract_text
from analysis import TextAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pdfminer

class PDFAnalyzerApp:
    def __init__(self):
        st.title("PDF Word Cloud Generator and Text Analysis")

    @staticmethod
    def upload_pdf():
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        return uploaded_file

    @staticmethod
    def display_entities(analyzer):
        st.subheader("Named Entity Recognition:")
        entities = analyzer.named_entity_recognition()
        st.write("Entities:", entities)

    @staticmethod
    def display_language(analyzer):
        st.subheader("Language Detection:")
        language = analyzer.language_detection()
        st.write("Language:", language)

    @staticmethod
    def display_pos_tags(analyzer):
        st.subheader("Part-of-Speech Tagging:")
        pos_tags = analyzer.part_of_speech_tagging()
        st.write("POS Tags:", pos_tags)

    @staticmethod
    def display_dependencies(analyzer):
        st.subheader("Dependency Parsing:")
        dependencies = analyzer.dependency_parsing()
        st.write("Dependencies:", dependencies)

    @staticmethod
    def display_sentiment_analysis(analyzer, pdf_text):
        st.subheader("Sentiment Analysis Results:")
        st.write("Sentiment Score:", analyzer.sentiment_analyzer.polarity_scores(pdf_text)['compound'])

    @staticmethod
    def display_selected_text_sentiment(analyzer):
        selected_text = st.text_area("Enter text for sentiment analysis:", value="")
        if selected_text:
            st.write("Sentiment Score:", analyzer.get_sentiment_score(selected_text))

    @staticmethod
    def display_metric_filter(analyzer):
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

    @staticmethod
    def display_keywords(analyzer):
        st.subheader("Keywords:")
        num_keywords = st.slider("Number of Keywords to Display", min_value=1, max_value=20, value=5)
        keywords = analyzer.extract_keywords(num_keywords)
        sort_option = st.radio("Sort by", ["Occurrence", "Alphabetically"])
        if sort_option == "Occurrence":
            keywords.sort(key=lambda x: x[1], reverse=True)
        else:
            keywords.sort(key=lambda x: x[0])
        st.write("Top Keywords:", keywords)

    @staticmethod
    def display_word_cloud(pdf_text):
        st.subheader("Word Cloud:")
        max_words = st.slider("Maximum Words in Word Cloud", min_value=50, max_value=500, value=150, step=50)
        background_color = st.color_picker("Background Color", "#FFFFFF")
        word_cloud = WordCloud(width=800, height=400, max_words=max_words, background_color=background_color).generate(pdf_text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud from PDF Text')
        st.pyplot(fig)

    @staticmethod
    def display_search_text(analyzer):
        st.subheader("Search Text:")
        search_query = st.text_input("Enter text to search")
        if search_query:
            search_results = analyzer.search_text(search_query)
            st.write("Search Results:", search_results)

    @staticmethod
    def display_sentence_analysis(analyzer):
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

    def run(self):
        uploaded_file = self.upload_pdf()
        if uploaded_file is not None:
            pdf_text = extract_text(uploaded_file)
            analyzer = TextAnalyzer(pdf_text)
            self.display_entities(analyzer)
            self.display_language(analyzer)
            self.display_pos_tags(analyzer)
            self.display_dependencies(analyzer)
            self.display_sentiment_analysis(analyzer, pdf_text)
            self.display_selected_text_sentiment(analyzer)
            self.display_metric_filter(analyzer)
            self.display_keywords(analyzer)
            self.display_word_cloud(pdf_text)
            self.display_search_text(analyzer)
            self.display_sentence_analysis(analyzer)


if __name__ == "__main__":
    app = PDFAnalyzerApp()
    app.run()
