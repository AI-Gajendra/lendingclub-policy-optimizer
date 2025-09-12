import argparse
import pandas as pd
import torch
import d3rlpy
from src.dl_model import LoanDefaultClassifier
from src.utils import load_config

def main(args):
    print("--- Starting Policy Comparison ---")

    # 1. Load Configurations
    dl_config = load_config('configs/dl_model_config.yaml')
    rl_config = load_config('configs/rl_agent_config.yaml')

    # 2. Load Test Data
    print("Loading test data (scaled and unscaled)...")

    test_scaled_df = pd.read_csv(dl_config['test_data_path'])
    X_test_scaled = test_scaled_df.drop(dl_config['target_column'], axis=1)
    
    test_unscaled_df = pd.read_csv(rl_config['test_data_path'])

    # 3. Load Trained Deep Learning Model
    print("Loading Deep Learning model...")
    input_features = X_test_scaled.shape[1]
    dl_model = LoanDefaultClassifier(
        input_features=input_features,
        hidden_layers=dl_config['hidden_layers'],
        dropout_rate=dl_config['dropout_rate']
    )
    dl_model.load_state_dict(torch.load(dl_config['model_save_path']))
    dl_model.eval()

    # 4. Load Trained Reinforcement Learning Agent
    print("Loading Reinforcement Learning agent...")
    rl_agent = d3rlpy.load_learnable(rl_config['model_save_path'], device='cpu')

    # 5. Get Predictions from Both Models
    print("Generating predictions from both models...")

    # DL Model Predictions
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
        probabilities = torch.sigmoid(dl_model(X_test_tensor)).numpy().flatten()
    
    # DL Policy: Approve (1) if predicted default probability is BELOW the threshold, otherwise Deny (0)
    dl_decisions = (probabilities < args.dl_threshold).astype(int)
    
    # RL Agent Predictions
    # The agent directly outputs the action (0 for Deny, 1 for Approve)
    rl_decisions = rl_agent.predict(X_test_scaled.to_numpy())

    # 6. Find and Analyze Conflicting Decisions
    print("Finding cases where the policies disagree...")
    
    analysis_df = test_unscaled_df.copy()
    analysis_df['dl_default_probability'] = probabilities
    analysis_df['dl_decision'] = dl_decisions
    analysis_df['rl_decision'] = rl_decisions

    conflicts = analysis_df[analysis_df['dl_decision'] != analysis_df['rl_decision']]

    output_path = "outputs/reports/conflicting_decisions.csv"
    print(f"Found {len(conflicts)} conflicting decisions.")
    print(f"Saving examples to {output_path}")
    
    conflicts.to_csv(output_path, index=False)
    
    rl_approves_dl_denies = conflicts[conflicts['rl_decision'] == 1]
    print("\n--- Example: RL Agent Approves, DL Model Denies ---")
    print("(These are applicants the RL agent considers profitable gambles despite higher risk)")
    
    rel_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'dl_default_probability', 'dl_decision', 'rl_decision']
    print(rl_approves_dl_denies[rel_cols].head(10))

    print("\n--- Policy Comparison Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare the decisions of DL and RL models.")
    parser.add_argument('--dl-threshold', type=float, default=0.3,
                        help="Probability threshold for the DL model to approve a loan. Approve if prob < threshold.")
    
    args = parser.parse_args()
    main(args)
    