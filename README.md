# Policy Optimization for Financial Decision-Making

## Project Overview

This project tackles a classic fintech problem: how should we decide whether to approve a loan?

The standard approach is to build a model that predicts the *probability of default*. But just predicting risk isn't the same as maximizing profit. An applicant might be slightly risky, but if their interest rate is high enough, approving the loan could still be a great financial decision.

That's the core idea here. We're comparing two different ways of thinking about this problem:

1.  **A Classic Predictive Model:** We build a standard Deep Learning (DL) classifier (an MLP) that's trained to do one thing: predict how likely someone is to default. Decisions are then made based on a simple risk threshold.
2.  **A Profit-Driven RL Agent:** We use offline Reinforcement Learning (specifically, Conservative Q-Learning) to train an agent on historical loan data. Its goal isn't to predict anything--it's to learn a direct approval/denial *policy* that maximizes the company's net profit.

The whole point is to build both, see how they behave differently, and really dig into the gap between a system that just avoids risk versus one that actively seeks to maximize financial returns.

---

## Project Structure

The repository is organized to be straightforward and reproducible. Here is the layout:

```
lendingclub-policy-optimizer/
├── configs/                  # YAML files for all the hyperparameters and paths
│   ├── dl_model_config.yaml
│   └── rl_agent_config.yaml
├── data/
│   ├── raw/                  # Where you'll drop the original LendingClub dataset
│   └── processed/            # Cleaned, model-ready data lands here
├── notebooks/                # Just a notebook for some initial EDA
│   └── 1_eda.ipynb
├── outputs/
│   ├── models/               # Trained models get saved here
│   └── reports/              # Evaluation metrics and analysis files
├── src/                      # All the reusable source code
│   ├── data_processing.py
│   ├── dl_model.py
│   └── utils.py
├── preprocess.py             # Script to run the full data cleaning pipeline
├── train.py                  # Script to train either the DL or RL model
├── evaluate.py               # Script to evaluate our trained models
├── compare_policies.py       # A script to analyze where the two models disagree
├── README.md                 # You're reading it!
└── requirements.txt          # All the project dependencies
```

---

## Getting Set Up

To execute the project, follow these steps.

**1. Clone the Repo**

```bash
git clone https://github.com/AI-Gajendra/lendingclub-policy-optimizer.git
cd lendingclub-policy-optimizer
```

**2. Set up a Virtual Environment (Highly Recommended)**

This keeps your project dependencies tidy.

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

**3. Install the Dependencies**

All the necessary libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

**4. Grab the Dataset**

You'll need to download the `accepted_2007_to_2018.csv` file from the [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Once you have it, just drop it inside the `data/raw/` directory.

---

## The Workflow: From Raw Data to Insights

This project is built around a command-line workflow. Just run the scripts in this order.

### Step 1: Prepping the Data

First things first, we need to clean up the raw data and get it ready for our models. This script handles all of that.

*   **Run it like this:**
    ```bash
    python preprocess.py
    ```
*   **What it does:**
    It takes the raw CSV, filters for loans that are actually completed (either 'Fully Paid' or 'Charged Off'), and then goes through a pretty standard cleaning process: dropping leaky or redundant columns, formatting features, imputing missing values, and one-hot encoding categorical data.
*   **What you get:**
    - `data/processed/train_scaled.csv` & `test_scaled.csv`: Normalized data, which is what the DL model needs.
    - `data/processed/train_unscaled.csv` & `test_unscaled.csv`: The same data, but kept at its original scale. We need this for calculating financial rewards for the RL agent and for our final analysis.
    - `data/processed/scaler.pkl`: The `StandardScaler` object that was fitted, so we can use it later.

### Step 2: Training the Models

Now we can train either the DL model or the RL agent.

*   **To train the Deep Learning model:**
    ```bash
    python train.py --model-type dl --config configs/dl_model_config.yaml
    ```
*   **To train the Reinforcement Learning agent:**
    ```bash
    python train.py --model-type rl --config configs/rl_agent_config.yaml
    ```
*   **Key Arguments:**
    - `--model-type [dl|rl]`: **(Required)** You have to tell it which model you want to train.
    - `--config [path]`: **(Required)** Points to the YAML file with all the model's hyperparameters.
*   **What you get:**
    - `outputs/models/deep_learning_classifier.pth`: The saved weights for the DL model.
    - `outputs/models/offline_rl_agent.d3`: The saved d3rlpy object for our trained RL agent.

### Step 3: Evaluating Model Performance

Once a model is trained, let's see how well it actually performs.

*   **To evaluate the Deep Learning model:**
    ```bash
    python evaluate.py --model-type dl --config configs/dl_model_config.yaml
    ```
*   **To evaluate the Reinforcement Learning agent:**
    ```bash
    python evaluate.py --model-type rl --config configs/rl_agent_config.yaml
    ```
*   **What it's doing:**
    - For the **DL model**, this is pretty standard. It loads the test set and calculates the **AUC** and **F1-Score** to see how good it is at classifying defaulters.
    - For the **RL agent**, things are different. We use Offline Policy Evaluation (OPE) to get an **Estimated Policy Value**. This is the cool part--it gives us an estimate of the average financial return we'd get per loan if we deployed this agent's policy.
*   **What you get:**
    - `outputs/reports/dl_metrics.json`: A JSON file with the DL model's scores.
    - `outputs/reports/rl_policy_value.json`: A JSON file with the RL agent's estimated dollar value.

### Step 4: Comparing the Two Policies

This is where significant insights are derived. This script is designed to find and analyze cases where our two models would make completely different decisions.

*   **Run it like this:**
    ```bash
    python compare_policies.py --dl-threshold 0.3
    ```
*   **What it's doing:**
    It loads both trained models and runs every applicant in the test set through them. The DL model's decision is based on the `--dl-threshold` you provide (it approves if the predicted default probability is *less than* the threshold). The RL agent just gives its direct policy decision. The script then flags every single case where their decisions conflict.
*   **What you get:**
    - `outputs/reports/conflicting_decisions.csv`: A CSV file containing the full data for every applicant the models disagreed on. This is perfect for a deep-dive analysis.
    - The script also prints a quick summary of interesting disagreements right to the console.

---

## So, What Did We Find?

*   **The DL model is a decent risk classifier.** It landed an **AUC of ~0.73**, which means it's pretty capable of distinguishing between a good and a bad applicant. The catch? With a standard 0.5 threshold, its **F1-Score was awful (~0.14)**, meaning it failed to identify most of the people who would actually default.

*   **The RL agent's policy looks very profitable.** It achieved an **Estimated Policy Value of ~$1,916**. This isn't an abstract score; it's a direct estimate of the average profit per loan. This shows that the agent learned a policy that is tightly aligned with the actual business goal.

*   **The real story is in the policy comparison.** This is where the RL agent's smarter strategy became obvious. It learned to approve some applicants that the DL model considered "high-risk," but only when the interest rate was high enough to make that risk worth taking. On the flip side, it also learned to deny some "safe" applicants if their interest rate was too low to be profitable.

In summary, the RL agent acts less like a simple risk analyst and more like a savvy portfolio manager, which represents a powerful shift in strategy.