# evaluate.py

import argparse
import yaml
import os
import json
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.dl_model import LoanDefaultClassifier
from src.utils import load_config
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import InitialStateValueEstimationEvaluator

def evaluate_dl_model(config):
    print("--- Starting Deep Learning Model Evaluation ---")
    print(f"Loading test data from {config['test_data_path']}...")
    test_df = pd.read_csv(config['test_data_path'])
    X_test = test_df.drop(config['target_column'], axis=1)
    y_test = test_df[config['target_column']]
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    print(f"Loading model from {config['model_save_path']}...")
    input_features = X_test.shape[1]
    model = LoanDefaultClassifier(input_features=input_features, hidden_layers=config['hidden_layers'], dropout_rate=config['dropout_rate'])
    try:
        model.load_state_dict(torch.load(config['model_save_path']))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config['model_save_path']}. Please train the model first.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Making predictions on the test set...")
    all_preds_proba = []
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        outputs = model(X_test_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        all_preds_proba.extend(probabilities.flatten())
    all_preds_binary = [1 if proba > 0.5 else 0 for proba in all_preds_proba]
    print("Calculating metrics...")
    auc = roc_auc_score(y_test, all_preds_proba)
    f1 = f1_score(y_test, all_preds_binary)
    precision = precision_score(y_test, all_preds_binary)
    recall = recall_score(y_test, all_preds_binary)
    metrics = {'AUC': auc, 'F1-Score': f1, 'Precision': precision, 'Recall': recall}
    print("\n--- Evaluation Results ---")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("--------------------------\n")
    print(f"Saving metrics to {config['metrics_save_path']}...")
    os.makedirs(os.path.dirname(config['metrics_save_path']), exist_ok=True)
    with open(config['metrics_save_path'], 'w') as f:
        json.dump(metrics, f, indent=4)
    print("--- DL Model Evaluation Finished ---")

def evaluate_rl_agent(config):
    """
    Loads a trained RL agent and evaluates its policy value.
    """
    print("--- Starting Offline RL Agent Evaluation ---")

    # 1. Load Data
    print("Loading test data...")
    test_scaled_df = pd.read_csv(config['test_data_path_scaled'])
    test_unscaled_df = pd.read_csv(config['test_data_path'])
    observations = test_scaled_df.drop(config['target_column'], axis=1).to_numpy()
    
    # 2. Reconstruct the test dataset
    num_samples = len(test_unscaled_df)
    actions = pd.Series([1] * num_samples)
    loan_amount_col = config['reward_columns']['loan_amount']
    int_rate_col = config['reward_columns']['interest_rate']
    target_col = config['target_column']
    test_unscaled_df[int_rate_col] = test_unscaled_df[int_rate_col] / 100.0
    rewards = test_unscaled_df.apply(lambda row: row[loan_amount_col] * row[int_rate_col] if row[target_col] == 0 else -row[loan_amount_col], axis=1)
    terminals = pd.Series([True] * num_samples)
    dataset = MDPDataset(observations=observations.astype('float32'), actions=actions.to_numpy().astype('int32'), rewards=rewards.to_numpy().astype('float32'), terminals=terminals.to_numpy())
    
    # 3. Load the Trained Agent
    print(f"Loading trained agent from {config['model_save_path']}...")
    try:
        device_str = 'cpu' if config['gpu'] < 0 else f"cuda:{config['gpu']}"
        rl_agent = d3rlpy.load_learnable(config['model_save_path'], device=device_str)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config['model_save_path']}. Please train the model first.")
        return

    # 4. Perform Offline Policy Evaluation using the correct Class
    print("Estimating policy value...")
    scorer = InitialStateValueEstimationEvaluator()
    policy_value = scorer(rl_agent, dataset)

    metrics = {'Estimated_Policy_Value': policy_value}
    
    print("\n--- Evaluation Results ---")
    print(f"Estimated Policy Value (Avg. Return per Loan): ${policy_value:.2f}")
    print("--------------------------\n")

    metrics_save_path = config.get('metrics_save_path', "outputs/reports/rl_policy_value.json")
    print(f"Saving metrics to {metrics_save_path}...")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("--- Offline RL Agent Evaluation Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--model-type', type=str, required=True, choices=['dl', 'rl'], help="Type of model to evaluate: 'dl' or 'rl'.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model's configuration YAML file.")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.model_type == 'dl':
        evaluate_dl_model(config)
    elif args.model_type == 'rl':
        evaluate_rl_agent(config)
    else:
        print("Invalid model type specified.")

if __name__ == '__main__':
    main()