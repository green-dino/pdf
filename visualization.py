import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

def interactive_sentiment_analysis(analyzer, selected_text, st):
    if st.checkbox("Interactive Sentiment Analysis"):
        sentiment_score = analyzer.sentiment_analyzer.polarity_scores(selected_text)['compound']
        st.write("Sentiment Score:", sentiment_score)

def generate_word_cloud(text, st):
    wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud from PDF Text')
    st.pyplot(fig)

def calculate_and_visualize_metrics(text, metric_filter, st):
    if metric_filter == "Word Count":
        words = word_tokenize(text)
        count = len(words)
        st.write(f"{metric_filter}: {count}")

    elif metric_filter == "Sentence Length":
        sentences = sent_tokenize(text)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        st.bar_chart(sentence_lengths)
        st.write(f"{metric_filter} Distribution")

    elif metric_filter == "Paragraph Length":
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_lengths = [len(word_tokenize(paragraph)) for paragraph in paragraphs]
        st.bar_chart(paragraph_lengths)
        st.write(f"{metric_filter} Distribution")

def analyze_paragraph_structure(paragraph, st):
    sentences = sent_tokenize(paragraph)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    relationships = []
    for i, sentence_tokens in enumerate(tokenized_sentences):
        if i < len(tokenized_sentences) - 1:
            relationships.append((sentence_tokens[-1], tokenized_sentences[i + 1][0]))

    if relationships:
        df = pd.DataFrame(relationships, columns=['Current Word', 'Next Word'])
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(pd.crosstab(df['Current Word'], df['Next Word']), cmap="YlGnBu", annot=True, fmt='g')
        plt.title('Relationships Between Tokens')
        st.pyplot(heatmap.figure)
