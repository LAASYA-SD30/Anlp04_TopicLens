
from flask import Flask, render_template, jsonify
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import re

app = Flask(__name__)

NUM_TOPICS = 5
NUM_TOP_WORDS = 10
GIBBS_ITERATIONS = 100
ALPHA = 0.1
BETA = 0.01
RANDOM_STATE = 42

# Sample document collection
DOCUMENTS = [
    "Artificial intelligence and machine learning are transforming modern technology.",
    "Deep learning models use neural networks to identify patterns in data.",
    "Natural language processing helps computers understand human language.",
    "Data science combines statistics programming and machine learning.",
    "Artificial intelligence is used in robotics automation and computer vision.",
    "Cloud computing provides scalable storage and computing resources.",
    "Cybersecurity protects computer networks from attacks and threats.",
    "Businesses use data analytics to improve decision making.",
    "Digital marketing uses social media advertising and customer analytics.",
    "Companies use machine learning to predict sales and customer behaviour.",
    "Financial institutions use data analysis for fraud detection.",
    "Stock markets respond to economic conditions and investor decisions.",
    "Business intelligence dashboards help managers track performance.",
    "Football teams use performance statistics to improve their strategies.",
    "Cricket players practise batting bowling and fielding techniques.",
    "Athletes need regular training nutrition and adequate recovery.",
    "Sports analytics helps coaches evaluate player performance.",
    "Exercise improves physical fitness and cardiovascular health.",
    "A balanced diet contains proteins vitamins minerals and carbohydrates.",
    "Doctors use medical imaging to identify diseases.",
    "Public health programmes help prevent infectious diseases.",
    "Climate change causes rising temperatures and extreme weather.",
    "Renewable energy sources include solar wind and hydropower.",
    "Air pollution affects human health and the environment.",
    "Forests protect biodiversity and support ecological balance.",
    "Students learn mathematics science programming and engineering.",
    "Online education provides digital learning resources to students.",
    "Universities encourage research innovation and practical learning.",
    "Libraries provide books journals and academic resources."
]


# --------------------------------------------------
# 1. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)


# --------------------------------------------------
# 2. DOCUMENT-TERM MATRIX
# --------------------------------------------------

def build_document_term_matrix(cleaned_documents):
    vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )

    dtm = vectorizer.fit_transform(cleaned_documents)
    vocabulary = vectorizer.get_feature_names_out()

    return vectorizer, dtm, vocabulary


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def normalize_rows(matrix):
    totals = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(totals, 1e-12)


def infer_topic_label(words):
    groups = {
        "Technology & AI": {
            "artificial", "intelligence", "machine", "learning",
            "technology", "data", "models", "networks", "computer",
            "neural", "cloud", "cybersecurity", "robotics"
        },
        "Business & Finance": {
            "businesses", "business", "financial", "markets", "stock",
            "sales", "companies", "marketing", "analytics", "customer",
            "fraud", "investor", "economic"
        },
        "Sports & Fitness": {
            "football", "cricket", "athletes", "sports", "players",
            "training", "exercise", "coaches", "batting", "bowling",
            "fitness", "recovery"
        },
        "Health & Medicine": {
            "health", "doctors", "medical", "diseases", "nutrition",
            "proteins", "vitamins", "infectious", "diet", "human"
        },
        "Environment & Energy": {
            "climate", "energy", "renewable", "solar", "wind",
            "pollution", "forests", "biodiversity", "environment",
            "temperatures", "ecological", "weather"
        },
        "Education & Research": {
            "students", "education", "universities", "research",
            "libraries", "academic", "mathematics", "science",
            "engineering", "journals"
        }
    }

    scores = {
        label: len(set(words) & group)
        for label, group in groups.items()
    }

    best_label = max(scores, key=scores.get)

    if scores[best_label] == 0:
        return "General Topic"

    return best_label


def extract_topics(topic_word_distributions, vocabulary):
    topics = []

    for topic_index, probabilities in enumerate(
        topic_word_distributions
    ):
        top_indices = np.argsort(probabilities)[::-1][:NUM_TOP_WORDS]

        words = [
            {
                "word": str(vocabulary[index]),
                "probability": float(probabilities[index])
            }
            for index in top_indices
        ]

        top_words = [item["word"] for item in words]

        topics.append({
            "topic_id": topic_index + 1,
            "words": words,
            "top_words": top_words,
            "label": infer_topic_label(top_words)
        })

    return topics


# --------------------------------------------------
# 4. VARIATIONAL BAYES LDA
# --------------------------------------------------

def run_variational_bayes_lda(dtm, vocabulary):
    model = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        max_iter=30,
        learning_method="batch",
        random_state=RANDOM_STATE,
        doc_topic_prior=1.0 / NUM_TOPICS,
        topic_word_prior=0.01
    )

    document_topic_distributions = model.fit_transform(dtm)

    topic_word_distributions = normalize_rows(model.components_)

    topics = extract_topics(
        topic_word_distributions,
        vocabulary
    )

    return {
        "model": model,
        "topics": topics,
        "topic_word_distributions":
            topic_word_distributions.tolist(),
        "document_topic_distributions":
            document_topic_distributions.tolist()
    }


# --------------------------------------------------
# 5. COLLAPSED GIBBS SAMPLING
# --------------------------------------------------

def run_gibbs_sampling(dtm, vocabulary):
    matrix = dtm.toarray().astype(int)
    num_documents, vocab_size = matrix.shape

    rng = np.random.default_rng(RANDOM_STATE)

    doc_ids = []
    word_ids = []

    # Expand counts into individual word tokens.
    for doc_id in range(num_documents):
        for word_id in range(vocab_size):
            for _ in range(matrix[doc_id, word_id]):
                doc_ids.append(doc_id)
                word_ids.append(word_id)

    doc_ids = np.asarray(doc_ids, dtype=np.int32)
    word_ids = np.asarray(word_ids, dtype=np.int32)

    num_tokens = len(word_ids)

    if num_tokens == 0:
        raise ValueError("No word tokens were found in the corpus.")

    document_topic_counts = np.zeros(
        (num_documents, NUM_TOPICS), dtype=np.int32
    )

    topic_word_counts = np.zeros(
        (NUM_TOPICS, vocab_size), dtype=np.int32
    )

    topic_counts = np.zeros(NUM_TOPICS, dtype=np.int32)

    assignments = rng.integers(
        0, NUM_TOPICS, size=num_tokens
    )

    # Random initialization of topic assignments.
    for i in range(num_tokens):
        d = doc_ids[i]
        w = word_ids[i]
        z = assignments[i]

        document_topic_counts[d, z] += 1
        topic_word_counts[z, w] += 1
        topic_counts[z] += 1

    # Repeated conditional sampling.
    for _ in range(GIBBS_ITERATIONS):
        for i in rng.permutation(num_tokens):
            d = doc_ids[i]
            w = word_ids[i]
            old_topic = assignments[i]

            # Remove current assignment.
            document_topic_counts[d, old_topic] -= 1
            topic_word_counts[old_topic, w] -= 1
            topic_counts[old_topic] -= 1

            # Conditional topic probabilities.
            document_factor = (
                document_topic_counts[d] + ALPHA
            )

            word_factor = (
                topic_word_counts[:, w] + BETA
            ) / (
                topic_counts + vocab_size * BETA
            )

            probabilities = document_factor * word_factor
            probabilities = probabilities / probabilities.sum()

            new_topic = rng.choice(
                NUM_TOPICS, p=probabilities
            )

            assignments[i] = new_topic

            # Add the new assignment.
            document_topic_counts[d, new_topic] += 1
            topic_word_counts[new_topic, w] += 1
            topic_counts[new_topic] += 1

    # Estimate topic-word probabilities.
    topic_word_distributions = (
        topic_word_counts + BETA
    ) / (
        topic_counts[:, None] + vocab_size * BETA
    )

    # Estimate document-topic probabilities.
    document_topic_distributions = (
        document_topic_counts + ALPHA
    ) / (
        document_topic_counts.sum(axis=1, keepdims=True)
        + NUM_TOPICS * ALPHA
    )

    topics = extract_topics(
        topic_word_distributions,
        vocabulary
    )

    return {
        "topics": topics,
        "topic_word_distributions":
            topic_word_distributions.tolist(),
        "document_topic_distributions":
            document_topic_distributions.tolist(),
        "iterations": GIBBS_ITERATIONS
    }


# --------------------------------------------------
# 6. TOPIC INTERPRETATION REPORT
# --------------------------------------------------

def generate_interpretation_report(
    vb_results,
    gibbs_results,
    document_count,
    vocabulary_size
):
    report = [
        {
            "heading": "1. Corpus and preprocessing",
            "text": (
                f"The corpus contains {document_count} documents and "
                f"{vocabulary_size} unique terms after preprocessing. "
                "Text was lowercased, URLs and punctuation were removed, "
                "and English stopwords and short tokens were filtered."
            )
        },
        {
            "heading": "2. Document-term representation",
            "text": (
                "CountVectorizer converts the cleaned corpus into a "
                "document-term matrix. Rows represent documents, columns "
                "represent vocabulary terms, and cell values represent "
                "term frequencies."
            )
        },
        {
            "heading": "3. Variational Bayes LDA",
            "text": (
                "Scikit-learn LDA uses variational inference to approximate "
                "the posterior distributions of latent topics. Its "
                "topic-word probabilities indicate which terms are most "
                "representative of each topic."
            ),
            "topics": [
                {
                    "label": topic["label"],
                    "top_words": ", ".join(topic["top_words"][:7])
                }
                for topic in vb_results["topics"]
            ]
        },
        {
            "heading": "4. Collapsed Gibbs Sampling",
            "text": (
                f"The custom Gibbs sampler performed {GIBBS_ITERATIONS} "
                "iterations. It repeatedly sampled word-topic assignments "
                "using document-topic counts, topic-word counts, and "
                "Dirichlet prior parameters."
            ),
            "topics": [
                {
                    "label": topic["label"],
                    "top_words": ", ".join(topic["top_words"][:7])
                }
                for topic in gibbs_results["topics"]
            ]
        },
        {
            "heading": "5. Comparing both methods",
            "text": (
                "Variational Bayes estimates an approximate posterior, "
                "whereas Gibbs Sampling uses repeated conditional sampling "
                "of latent topic assignments. Topic numbering is arbitrary, "
                "so compare top words and semantic themes rather than "
                "matching topics by their numeric IDs."
            )
        },
        {
            "heading": "6. Limitations",
            "text": (
                "The demonstration uses a small, predefined corpus and "
                "five topics. The number of topics and Gibbs iterations "
                "may affect the results. The displayed topic labels are "
                "heuristic descriptions based on the top words and should "
                "be checked by a human reader."
            )
        }
    ]

    return report


# --------------------------------------------------
# 7. FLASK DASHBOARD
# --------------------------------------------------

@app.route("/")
def index():
    cleaned_documents = [
        preprocess_text(document) for document in DOCUMENTS
    ]

    vectorizer, dtm, vocabulary = build_document_term_matrix(
        cleaned_documents
    )

    vb_results = run_variational_bayes_lda(dtm, vocabulary)
    gibbs_results = run_gibbs_sampling(dtm, vocabulary)

    document_summaries = []

    for i, document in enumerate(DOCUMENTS):
        vb_distribution = vb_results[
            "document_topic_distributions"
        ][i]

        gibbs_distribution = gibbs_results[
            "document_topic_distributions"
        ][i]

        vb_index = int(np.argmax(vb_distribution))
        gibbs_index = int(np.argmax(gibbs_distribution))

        document_summaries.append({
            "document_id": i + 1,
            "original_text": document,
            "cleaned_text": cleaned_documents[i],
            "dominant_topic": vb_results["topics"][vb_index]["label"],
            "topic_id": vb_index + 1,
            "topic_probability": float(vb_distribution[vb_index]),
            "gibbs_dominant_topic":
                gibbs_results["topics"][gibbs_index]["label"],
            "gibbs_topic_id": gibbs_index + 1,
            "gibbs_topic_probability":
                float(gibbs_distribution[gibbs_index])
        })

    # Preview the first 12 vocabulary terms.
    preview_columns = min(12, len(vocabulary))
    dense_preview = dtm[:, :preview_columns].toarray()

    dtm_preview = [
        {
            "document_id": row_index + 1,
            "values": row.tolist()
        }
        for row_index, row in enumerate(dense_preview)
    ]

    dtm_terms = vocabulary[:preview_columns].tolist()

    stats = {
        "document_count": len(DOCUMENTS),
        "vocabulary_size": len(vocabulary),
        "topic_count": NUM_TOPICS,
        "token_count": int(dtm.sum()),
        "gibbs_iterations": GIBBS_ITERATIONS,
        "dtm_rows": int(dtm.shape[0]),
        "dtm_columns": int(dtm.shape[1])
    }

    report = generate_interpretation_report(
        vb_results,
        gibbs_results,
        len(DOCUMENTS),
        len(vocabulary)
    )

    return render_template(
        "index.html",
        stats=stats,
        documents=DOCUMENTS,
        cleaned_documents=cleaned_documents,
        vocabulary=vocabulary.tolist(),
        dtm_terms=dtm_terms,
        dtm_preview=dtm_preview,
        vb_topics=vb_results["topics"],
        gibbs_topics=gibbs_results["topics"],
        topics=vb_results["topics"],
        document_summaries=document_summaries,
        report=report,
        vb_results=vb_results,
        gibbs_results=gibbs_results
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "application": "TopicLens",
        "methods": [
            "Variational Bayes LDA",
            "Collapsed Gibbs Sampling"
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)
