# train.py

import argparse
import yaml
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.dl_model import LoanDefaultClassifier
from src.utils import load_config
import d3rlpy
from d3rlpy.dataset import MDPDataset

def train_dl_model(config):
    print("--- Starting Deep Learning Model Training ---")

    # 1. Load Data
    print("Loading and preparing data...")
    train_df = pd.read_csv(config['train_data_path'])
    
    X_train = train_df.drop(config['target_column'], axis=1)
    y_train = train_df[config['target_column']]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 2. Initialize Model, Loss, and Optimizer
    input_features = X_train.shape[1]
    model = LoanDefaultClassifier(
        input_features=input_features,
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate']
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")

    # 3. Training Loop
    print("Starting training loop...")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}], Average Loss: {avg_loss:.4f}")

    # 4. Save the Model
    print(f"Saving trained model to {config['model_save_path']}...")
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['model_save_path'])

    print("--- DL Model Training Finished ---")


def train_rl_agent(config):
    """
    Prepares data and trains the Offline RL agent.
    """
    print("--- Starting Offline RL Agent Training ---")

    # 1. Load Data
    print("Loading scaled and unscaled data...")
    train_scaled_df = pd.read_csv(config['train_data_path_scaled'])
    train_unscaled_df = pd.read_csv(config['train_data_path'])
    
    observations = train_scaled_df.drop(config['target_column'], axis=1).to_numpy()
    
    # 2. Define Actions and Rewards
    num_samples = len(train_unscaled_df)
    actions = pd.Series([1] * num_samples)
    
    print("Calculating rewards...")
    loan_amount_col = config['reward_columns']['loan_amount']
    int_rate_col = config['reward_columns']['interest_rate']
    target_col = config['target_column']
    
    train_unscaled_df[int_rate_col] = train_unscaled_df[int_rate_col] / 100.0

    rewards = train_unscaled_df.apply(
        lambda row: row[loan_amount_col] * row[int_rate_col] if row[target_col] == 0 else -row[loan_amount_col],
        axis=1
    )
    
    # 3. Create d3rlpy Dataset
    terminals = pd.Series([True] * num_samples)
    
    dataset = MDPDataset(
        observations=observations.astype('float32'),
        actions=actions.to_numpy().astype('int32'),
        rewards=rewards.to_numpy().astype('float32'),
        terminals=terminals.to_numpy()
    )

    # 4. Initialize and Train the RL Agent
    print(f"Initializing {config['algorithm'].upper()} agent...")
    if config['algorithm'] == 'cql':
        rl_agent = d3rlpy.algos.DiscreteCQLConfig().create(device=config['gpu'] >= 0)
    else:
        raise NotImplementedError(f"Algorithm {config['algorithm']} is not implemented.")

    print(f"Starting training for {config['n_steps']} steps...")
    rl_agent.fit(
        dataset,
        n_steps=config['n_steps']
    )
    
    # 5. Save the RL Model
    print(f"Saving trained RL agent to {config['model_save_path']}...")
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    rl_agent.save(config['model_save_path'])

    print("--- Offline RL Agent Training Finished ---")



def main():
    parser = argparse.ArgumentParser(description="Train a model for loan default prediction.")
    parser.add_argument('--model-type', type=str, required=True, choices=['dl', 'rl'],
                        help="Type of model to train: 'dl' for Deep Learning or 'rl' for Reinforcement Learning.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the model's configuration YAML file.")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.model_type == 'dl':
        train_dl_model(config)
    elif args.model_type == 'rl':
        train_rl_agent(config)
    else:
        print("Invalid model type specified.")

if __name__ == '__main__':
    main()