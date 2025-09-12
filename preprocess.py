# preprocess.py

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.data_processing import clean_data

def main(args):
    print("Starting data preprocessing...")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading raw data from {args.input_path}...")
    try:
        df = pd.read_csv(args.input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The file {args.input_path} was not found.")
        return

    # 2. Clean and Feature Engineer Data
    print("Cleaning data and engineering features...")
    df_clean = clean_data(df)
    
    print(f"Data shape after cleaning: {df_clean.shape}")

    # 3. Split Data
    print("Splitting data into training and testing sets...")
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # 4. Scale Features
    print("Scaling numerical features...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # 5. Save Artifacts
    print("Saving processed data and scaler...")
    
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)
    
    # Save the original unscaled data as well, as it's needed for RL reward calculation
    train_df_unscaled = pd.concat([X_train, y_train], axis=1)
    test_df_unscaled = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(args.output_dir, 'train_scaled.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_scaled.csv'), index=False)
    train_df_unscaled.to_csv(os.path.join(args.output_dir, 'train_unscaled.csv'), index=False)
    test_df_unscaled.to_csv(os.path.join(args.output_dir, 'test_unscaled.csv'), index=False)
    
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))

    print("Preprocessing finished successfully.")
    print(f"Artifacts saved in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess LendingClub loan data.")
    parser.add_argument('--input-path', type=str, default='data/raw/accepted_2007_to_2018.csv',
                        help='Path to the raw CSV file.')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory to save the processed files.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to allocate to the test split.')
    
    args = parser.parse_args()
    main(args)
