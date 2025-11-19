import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def linkage_attack_tfidf(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "private_id",
) -> dict:
    """
    Linkage attack using TF-IDF cosine similarity.
    Attempts to link synthetic records to private records.
    """
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    all_texts = public_df[text_col].tolist() + private_df[text_col].tolist()
    vectorizer.fit(all_texts)

    # Transform texts
    public_vectors = vectorizer.transform(public_df[text_col])
    private_vectors = vectorizer.transform(private_df[text_col])

    # Compute similarities
    similarities = cosine_similarity(private_vectors, public_vectors)

    # For each private record, find the most similar public record
    most_similar_indices = similarities.argmax(axis=1)
    predicted_ids = public_df.iloc[most_similar_indices][id_col].values
    true_ids = private_df[id_col].values

    # Calculate accuracy
    correct = (predicted_ids == true_ids).sum()
    total = len(private_df)
    accuracy = float(correct / total) if total > 0 else 0.0

    return {
        "linkage_attack_tfidf/accuracy": accuracy,
        "linkage_attack_tfidf/nb_docs": total,
        "linkage_attack_tfidf/nb_correct": int(correct),
    }


def random_leakage_attack(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    id_col: str = "private_id",
) -> dict:
    """Random leakage attack - randomly assigns private IDs to measure baseline."""
    np.random.seed(42)  # For reproducible results
    
    true_ids = private_df[id_col].values
    unique_public_ids = public_df[id_col].unique()
    
    # Randomly predict IDs from the available public IDs
    predicted_ids = np.random.choice(unique_public_ids, size=len(true_ids), replace=True)
    
    acc = float(np.mean(predicted_ids == true_ids))
    return {
        "random_leakage/accuracy": acc,
        "random_leakage/nb_docs": len(private_df),
        "random_leakage/nb_correct": int(np.sum(predicted_ids == true_ids)),
    }


def proximity_attack_tfidf(
    synthetic_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "private_id",
    n_samples: int = 3,
) -> dict:
    """Proximity-based Membership Inference Attack."""
    np.random.seed(42)  # For reproducible results
    
    # Fit TF-IDF vectorizer on the full dataset
    all_texts = synthetic_df[text_col].tolist() + private_df[text_col].tolist()
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer.fit(all_texts)
    
    correct_predictions = 0
    total_predictions = 0
    
    # Get unique user IDs
    unique_users = private_df[id_col].unique()
    
    for synthetic_idx, synthetic_row in synthetic_df.iterrows():
        synthetic_text = synthetic_row[text_col]
        true_user_id = synthetic_row[id_col]
        
        # Skip if true user not in private documents
        if true_user_id not in unique_users:
            continue
            
        # Step 1: Select comparison documents
        same_user_docs = private_df[private_df[id_col] == true_user_id]
        different_user_docs = private_df[private_df[id_col] != true_user_id]
        
        if len(same_user_docs) == 0 or len(different_user_docs) == 0:
            continue
            
        # Sample documents
        n_same = min(n_samples, len(same_user_docs))
        n_diff = min(n_samples, len(different_user_docs))
        
        sampled_same = same_user_docs.sample(n=n_same, random_state=42)
        sampled_diff = different_user_docs.sample(n=n_diff, random_state=42)
        
        # Transform synthetic document once
        synthetic_vector = vectorizer.transform([synthetic_text])
        
        # Step 2: Compute similarities using the fitted vectorizer
        same_user_similarities = []
        for _, same_doc in sampled_same.iterrows():
            comparison_vector = vectorizer.transform([same_doc[text_col]])
            similarity = cosine_similarity(synthetic_vector, comparison_vector)[0][0]
            same_user_similarities.append(similarity)
        
        diff_user_similarities = []
        for _, diff_doc in sampled_diff.iterrows():
            comparison_vector = vectorizer.transform([diff_doc[text_col]])
            similarity = cosine_similarity(synthetic_vector, comparison_vector)[0][0]
            diff_user_similarities.append(similarity)
        
        # Compute mean similarities
        similarity_same = np.mean(same_user_similarities)
        similarity_random = np.mean(diff_user_similarities)
        
        # Step 3: Proximity test
        total_predictions += 1
        if similarity_same > similarity_random:
            # Predict "LEAKAGE" - synthetic document is closer to real user
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "proximity_attack_tfidf/accuracy": float(accuracy),
        "proximity_attack_tfidf/nb_docs": total_predictions,
        "proximity_attack_tfidf/nb_correct": correct_predictions,
    }


def proximity_attack_embeddings(
    synthetic_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "private_id",
    n_samples: int = 3,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> dict:
    """Proximity-based Membership Inference Attack using sentence embeddings."""
    np.random.seed(42)  # For reproducible results
    
    # Load embedding model
    model = SentenceTransformer(model_name)
    
    correct_predictions = 0
    total_predictions = 0
    
    # Get unique user IDs
    unique_users = private_df[id_col].unique()
    
    for synthetic_idx, synthetic_row in synthetic_df.iterrows():
        synthetic_text = synthetic_row[text_col]
        true_user_id = synthetic_row[id_col]
        
        # Skip if true user not in private documents
        if true_user_id not in unique_users:
            continue
            
        # Step 1: Select comparison documents
        same_user_docs = private_df[private_df[id_col] == true_user_id]
        different_user_docs = private_df[private_df[id_col] != true_user_id]
        
        if len(same_user_docs) == 0 or len(different_user_docs) == 0:
            continue
            
        # Sample documents
        n_same = min(n_samples, len(same_user_docs))
        n_diff = min(n_samples, len(different_user_docs))
        
        sampled_same = same_user_docs.sample(n=n_same, random_state=42)
        sampled_diff = different_user_docs.sample(n=n_diff, random_state=42)
        
        # Encode synthetic document once
        synthetic_embedding = model.encode([synthetic_text])
        
        # Step 2: Compute similarities using embeddings
        same_user_similarities = []
        for _, same_doc in sampled_same.iterrows():
            comparison_embedding = model.encode([same_doc[text_col]])
            similarity = cosine_similarity(synthetic_embedding, comparison_embedding)[0][0]
            same_user_similarities.append(similarity)
        
        diff_user_similarities = []
        for _, diff_doc in sampled_diff.iterrows():
            comparison_embedding = model.encode([diff_doc[text_col]])
            similarity = cosine_similarity(synthetic_embedding, comparison_embedding)[0][0]
            diff_user_similarities.append(similarity)
        
        # Compute mean similarities
        similarity_same = np.mean(same_user_similarities)
        similarity_random = np.mean(diff_user_similarities)
        
        # Step 3: Proximity test
        total_predictions += 1
        if similarity_same > similarity_random:
            # Predict "LEAKAGE" - synthetic document is closer to real user
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "proximity_attack_embeddings/accuracy": float(accuracy),
        "proximity_attack_embeddings/nb_docs": total_predictions,
        "proximity_attack_embeddings/nb_correct": correct_predictions,
    }


def proximity_attack_random(
    synthetic_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "text",
    id_col: str = "private_id",
    n_samples: int = 3,
) -> dict:
    """Proximity-based Membership Inference Attack using random choice (baseline)."""
    np.random.seed(42)  # For reproducible results
    
    correct_predictions = 0
    total_predictions = 0
    
    # Get unique user IDs
    unique_users = private_df[id_col].unique()
    
    for synthetic_idx, synthetic_row in synthetic_df.iterrows():
        true_user_id = synthetic_row[id_col]
        
        # Skip if true user not in private documents
        if true_user_id not in unique_users:
            continue
            
        # Step 1: Select comparison documents
        same_user_docs = private_df[private_df[id_col] == true_user_id]
        different_user_docs = private_df[private_df[id_col] != true_user_id]
        
        if len(same_user_docs) == 0 or len(different_user_docs) == 0:
            continue
        
        # Step 2: Random choice instead of similarity computation
        total_predictions += 1
        random_choice = np.random.choice([True, False])  # True = same user closer, False = different user closer
        
        if random_choice:
            # Randomly predict "LEAKAGE"
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "proximity_attack_random/accuracy": float(accuracy),
        "proximity_attack_random/nb_docs": total_predictions,
        "proximity_attack_random/nb_correct": correct_predictions,
    }