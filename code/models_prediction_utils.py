import pandas as pd

def predict_HuggingFace_models(model_name, model, tokenizer, samples, nb_samples, max_length):
    """
    Given a model, tokenizer, and samples, this function computes the predictions
    for each sample and returns the non-safe labels based on the model.

    Args:
    - tokenizer: The tokenizer for the model.
    - model: The trained HuggingFace model.
    - samples: The samples (e.g., text) for which predictions will be made.
    - nb_samples: The number of samples to process.
    - model_name: A string indicating which model is being used 
        - 'EthicalEye' 
        - 'KoalaAI'

    Returns:
    - A list of dictionaries containing predictions and non-safe probabilities.
    """
    
    # Define the non-safe labels based on the model
    if model_name == "EthicalEye":
        non_safe_labels = {"Un-Safe"}  # Labels for EthicalEye
    elif model_name == "KoalaAI":
        non_safe_labels = {'H', 'SH', 'V', 'S', 'HR', 'V2', 'S3', 'H2'}  # Labels for KoalaAI
    else:
        raise ValueError("Unsupported model name. Please choose either 'EthicalEye' or 'KoalaAI'.")
    
    # List to store predictions for each sample
    predictions = []
    
    # Iterate over each sample (we'll process a subset if nb_samples is provided)
    for idx, comment in enumerate(samples[:nb_samples]):
        # Tokenize the input
        inputs = tokenizer(comment, return_tensors="pt", max_length=max_length, truncation=True)

        # Get model outputs
        outputs = model(**inputs)
        
        # Get the predicted logits
        logits = outputs.logits
        
        # Apply softmax to get probabilities (scores)
        probabilities = logits.softmax(dim=-1).squeeze()
        
        # Retrieve the labels (assuming model.config.id2label exists)
        id2label = model.config.id2label
        labels = [id2label[idx] for idx in range(len(probabilities))]
        
        # Initialize the maximum probability for non-safe labels
        max_non_safe_prob = 0.0
        
        # Find the highest probability of non-safe label(s)
        for label, probability in zip(labels, probabilities):
            if label in non_safe_labels and probability.item() > max_non_safe_prob:
                max_non_safe_prob = probability.item()

        # Store the comment and non-safe probability
        predictions.append({"comment": comment, "non_safe_probability": max_non_safe_prob})
    
    return pd.DataFrame(predictions)


def predict_citizenlab_model(citizenlab_toxicity_classifier, samples, nb_samples, max_length=512):
    """
    Given the CitizenLab toxicity classifier and samples, this function computes the predictions
    for each sample and returns the non-safe labels.

    Args:
    - citizenlab_toxicity_classifier: The CitizenLab classifier function.
    - samples: The samples (e.g., text) for which predictions will be made.
    - nb_samples: The number of samples to process.

    Returns:
    - A list of dictionaries containing predictions and non-safe probabilities.
    """
    
    # Define the non-safe label for CitizenLab
    non_safe_labels = {"non-toxic"}
    
    # List to store predictions for each sample
    predictions = []
    
    # Iterate over each sample (we'll process a subset if nb_samples is provided)
    for idx, comment in enumerate(samples[:nb_samples]):
        # Use CitizenLab-specific classifier
        res = citizenlab_toxicity_classifier(comment, max_length=max_length, truncation=True)
        
        # Binary classifier with classes 'toxic' and 'non-toxic'
        non_safe_prob = 1 - res[0]['score']
        if res[0]['label'] in non_safe_labels:
            non_safe_prob = res[0]['score']
        
        # Store the comment and non-safe probability
        predictions.append({"comment": comment, "non_safe_probability": non_safe_prob})
    
    return pd.DataFrame(predictions)
