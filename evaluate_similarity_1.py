import pandas as pd

def evaluate_similarity(reference_tokens, predicted_tokens):
    """
    Calculate precision, recall, and F-score between two sets of tokens.

    Parameters:
    - reference_tokens: list of tokens from the reference document
    - predicted_tokens: list of tokens from the predicted document

    Returns:
    - precision: Precision score
    - recall: Recall score
    - fscore: F-score
    """
    reference_words = set(reference_tokens)
    predicted_words = set(predicted_tokens)
    
    common_words = reference_words.intersection(predicted_words)
    C = len(common_words)
    X_len = len(reference_words)
    X_hat_len = len(predicted_words)
    
    precision = C / X_hat_len if X_hat_len > 0 else 0
    recall = C / X_len if X_len > 0 else 0
    fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, fscore

def evaluate_datasets(test, result_1):
    """
    Evaluate precision, recall, and F-score across test and result datasets.

    Parameters:
    - test: DataFrame with test data containing 'Relevant_Documents_Tokenized' column
    - result_1: DataFrame with result data containing 'Relevant_Documents_Tokenized' column

    Returns:
    - precision_list: List of precision values
    - recall_list: List of recall values
    - fscore_list: List of F-score values
    - bad_cases_df: DataFrame containing the worst 40 cases where recall < 0.5
    """
    test_indices = range(0, len(test), 1)
    result_indices = range(0, len(result_1), 4)
    
    pairs = zip(test_indices, result_indices)
    precision_list, recall_list, fscore_list, bad_cases = [], [], [], []
    
    for test_idx, result_idx in pairs:
        if test_idx < len(test) and result_idx < len(result_1):
            test_tokens = test['Relevant_Documents_Tokenized'].iloc[test_idx]
            predicted_tokens = result_1['Relevant_Documents_Tokenized'].iloc[result_idx]
            
            precision, recall, fscore = evaluate_similarity(test_tokens, predicted_tokens)
            
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)

            # Store cases with recall < 0.5 for later filtering
            if recall < 0.7:
                bad_cases.append({
                    'Test Index': test_idx,
                    'Result Index': result_idx,
                    'Recall': recall,
                    'Precision': precision,
                    'F-score': fscore,
                    'Test Text': test['Relevant_Documents_Tokenized'].iloc[test_idx],
                    'Predicted Text': result_1['Relevant_Documents_Tokenized'].iloc[result_idx]
                })

    # Convert the list of bad cases into a DataFrame
    bad_cases_df = pd.DataFrame(bad_cases)

    # Sort by recall and pick the 40 worst cases
    worst_40_cases = bad_cases_df.sort_values(by='Recall').head(40)
    
    return precision_list, recall_list, fscore_list, worst_40_cases

def es1(test_file, result_file, output_file):
    """
    Function to load the datasets, evaluate similarity, and save the worst 40 cases with recall < 0.5.

    Parameters:
    - test_file: Path to the test CSV file
    - result_file: Path to the result CSV file
    - output_file: Path to save the worst 40 cases CSV

    Returns:
    - avg_precision: Average precision score
    - avg_recall: Average recall score
    - avg_fscore: Average F-score
    """
    # Load datasets
    test = pd.read_csv(test_file)
    result_1 = pd.read_csv(result_file)

    # Evaluate the datasets and get the worst 40 cases
    precision_list, recall_list, fscore_list, worst_40_cases = evaluate_datasets(test, result_1)

    # Save the worst 40 cases to a CSV file
    worst_40_cases.to_csv(output_file, index=False)

    # Calculate averages
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    avg_fscore = sum(fscore_list) / len(fscore_list) if fscore_list else 0

    # Return the averages
    return avg_precision, avg_recall, avg_fscore
