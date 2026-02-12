# Natural Language Processing - Basics

## What is Natural Language Processing?

Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language in a meaningful and useful way.

## Key Applications

- **Text Classification**: Categorize documents, emails, or social media posts
- **Sentiment Analysis**: Determine emotional tone of text
- **Machine Translation**: Translate text between languages
- **Named Entity Recognition**: Identify people, places, organizations
- **Question Answering**: Answer questions based on documents
- **Text Summarization**: Create concise summaries of documents
- **Chatbots**: Conversational AI systems
- **Speech Recognition**: Convert audio to text
- **Information Retrieval**: Search and find relevant documents

## Text Representation

### Raw Text to Numbers

Since machine learning models work with numbers, we need to convert text to numerical representations.

### Tokenization

Breaking text into individual tokens (words, subwords, or characters):

```
"Hello, world!" → ["Hello", ",", "world", "!"]
```

### Vocabulary

A vocabulary is the set of unique tokens:

$$
V = \{\text{word}_1, \text{word}_2, \ldots, \text{word}_n\}
$$

### One-Hot Encoding

Each word is represented as a vector with one 1 and rest 0s:

```
"hello" → [1, 0, 0, 0, ...]
"world" → [0, 0, 1, 0, ...]
```

Limitations: High dimensionality, no semantic meaning.

## Fundamental Concepts

### Bag of Words (BoW)

Represents a document as a collection of word frequencies, ignoring order:

$$
\text{BoW} = [count(\text{word}_1), count(\text{word}_2), \ldots, count(\text{word}_n)]
$$

### Term Frequency-Inverse Document Frequency (TF-IDF)

Weighs words by their importance:

$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)
$$

Where:
- $\text{TF}(t,d)$ = frequency of term $t$ in document $d$
- $\text{IDF}(t) = \log\left(\frac{N}{N_t}\right)$ = inverse document frequency

### N-grams

Sequences of n consecutive tokens:
- **Unigram** (n=1): Individual words
- **Bigram** (n=2): Two-word sequences
- **Trigram** (n=3): Three-word sequences

Example: "The cat sat on the mat"
- Bigrams: ["The cat", "cat sat", "sat on", ...]

## Word Embeddings

Modern approach representing words as dense vectors:

### Word2Vec

Maps words to $d$-dimensional vectors where similar words are close together:

$$
\text{word embeddings}: \mathbb{R}^{|V|} \rightarrow \mathbb{R}^d
$$

Advantages:
- Lower dimensionality than one-hot encoding
- Captures semantic relationships
- Can perform analogies: "king - man + woman ≈ queen"

### GloVe (Global Vectors)

Combines frequency statistics and distributional similarity to create word embeddings.

### FastText

Extension of Word2Vec using subword information, better for rare words and morphologically rich languages.

## NLP Pipeline

```
Raw Text
    ↓
Preprocessing (cleaning, lowercasing)
    ↓
Tokenization (split into words)
    ↓
Normalization (stemming, lemmatization)
    ↓
Feature Extraction (embeddings, TF-IDF)
    ↓
Model Training/Inference
    ↓
Post-processing
    ↓
Output (classification, translation, etc.)
```

### Preprocessing

1. **Lowercasing**: Convert to lowercase for consistency
2. **Removing punctuation**: Remove special characters
3. **Removing stopwords**: Remove common words (a, the, is, etc.)
4. **Stemming**: Reduce words to root form (running → run)
5. **Lemmatization**: Convert to base dictionary form

## NLP Tasks

### Text Classification
Assign text to predefined categories.
- Spam detection
- Sentiment analysis
- Topic classification

### Named Entity Recognition (NER)
Identify and classify named entities:
- People
- Organizations
- Locations
- Dates
- Products

### Machine Translation
Translate text from source to target language.

Modern approach: Sequence-to-sequence models with attention mechanisms.

### Question Answering
Answer questions based on documents or knowledge bases.

### Text Summarization
Generate concise summaries of long documents.

**Types:**
- **Extractive**: Select important sentences from original
- **Abstractive**: Generate new sentences capturing main ideas

### Sentiment Analysis
Determine emotion/opinion in text.

Example: "I love this product!" → Positive sentiment

### Machine Reading Comprehension
Answer questions based on given passage.

Example: Given passage about a book, answer questions about its content.

## Language Models

### What is a Language Model?

A language model predicts probability of next word given previous words:

$$
P(\text{word}_n | \text{word}_1, \text{word}_2, \ldots, \text{word}_{n-1})
$$

### Recurrent Neural Networks (RNNs)

Process sequences by maintaining hidden state:
- **LSTM** (Long Short-Term Memory): Better handles long-term dependencies
- **GRU** (Gated Recurrent Unit): Lighter version of LSTM

### Transformer Models

Modern architecture based on attention mechanisms:
- **BERT**: Bidirectional understanding
- **GPT**: Generative pre-trained models
- **T5**: Transfer transformer for various tasks

## Challenges in NLP

- **Ambiguity**: Words and phrases with multiple meanings
- **Context Dependency**: Meaning depends on context
- **Rare Words**: Limited training data for uncommon words
- **Multiple Languages**: Different grammar and structures
- **Sarcasm and Idioms**: Figurative language difficult to parse
- **Domain Adaptation**: Models trained on one domain may not work on another

## Evaluation Metrics

### Classification Tasks
- Accuracy
- Precision
- Recall
- F1-score

### Sequence-to-Sequence Tasks
- **BLEU Score**: Measures overlap with reference translation
- **ROUGE Score**: Measures overlap for summarization
- **Perplexity**: For language models

### NER and Parsing
- Precision, Recall, F1 (per entity type)
- Exact match accuracy

## Popular NLP Libraries

| Library | Purpose |
|---------|---------|
| NLTK | Tokenization, stemming, parsing |
| spaCy | NLP pipelines, NER |
| Transformers | Pre-trained models (BERT, GPT, etc.) |
| VADER | Sentiment analysis |
| Gensim | Topic modeling, word embeddings |
| TextBlob | Simple NLP tasks |

## Popular Datasets

- **GLUE**: General Language Understanding Evaluation
- **SQuAD**: Question answering
- **MNIST Papers**: Text classification
- **Universal Dependencies**: Parsing
- **WMT**: Machine translation

## Further Reading

- [Word Embeddings and Embedders](SimpleEmbedders.md)
- [Attention Mechanisms](Attention.md)
- [Transformers](Transformers.md)
