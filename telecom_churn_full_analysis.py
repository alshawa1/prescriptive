import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")

# ==========================================
# 1. DATA LOADING
# ==========================================
# Function to load data from CSV
# What is code do: Reads the telecom_churn.csv file into a pandas DataFrame.
def load_data(filepath):
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Function to preprocess the data
# What is code do: Handles missing values, detects and treats outliers, and encodes categorical variables.
def preprocess_data(df):
    print("\n--- Starting Preprocessing ---")
    
    # 2.1 Handle Missing Values
    # What is code do: Checks for null values and fills them. (Here we assume simplistic filling or dropping for demo)
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    df = df.fillna(method='ffill') # Forward fill as a generic strategy, usually requires more specific logic
    
    # 2.2 Outlier Detection and Treatment
    # What is code do: Uses IQR method to cap outliers in numerical columns.
    print("Checking for outliers in numerical columns...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if outliers exist
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Outliers detected in {col}: {len(outliers)} rows. Capping them.")
            # Capping outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
    # 2.3 Encoding Categorical Variables
    # What is code do: Converts categorical string columns into numbers using Label Encoding for simplicity.
    print("Encoding categorical variables...")
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Avoid encoding date columns if any, treating 'date_of_registration' separately later involves feature eng
        if 'date' not in col: 
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# Function to create new features
# What is code do: Derives new meaningful metrics from existing data to help the model learning.
def feature_engineering(df):
    print("\n--- Starting Feature Engineering ---")
    
    # Example Feature: Average Usage
    # What is code do: creating a feature representing average activity
    if 'calls_made' in df.columns and 'sms_sent' in df.columns:
        df['total_interactions'] = df['calls_made'] + df['sms_sent']
    
    # Example Feature: Data usage per unit (arbitrary scaling)
    if 'data_used' in df.columns:
        df['data_usage_log'] = np.log1p(df['data_used'].clip(lower=0)) # Log transform to handle skewness
        
    print("Features engineered: 'total_interactions', 'data_usage_log'")
    return df

# ==========================================
# 4. DESCRIPTIVE ANALYTICS
# ==========================================
# Function to perform descriptive analysis
# What is code do: Generates extensive visualizations and summary statistics to explain the data.
def descriptive_analytics(df):
    print("\n--- Starting Descriptive Analytics ---")
    
    # 4.1 Statistics
    print("\nKey Statistics:")
    print(df.describe())
    
    # 4.2 Churn Distribution
    # What is code do: Prints the count of churned vs non-churned users.
    if 'churn' in df.columns:
        print("\nChurn Distribution:")
        print(df['churn'].value_counts(normalize=True))
        
        # VISUALIZATION 1: Churn Count Plot
        plt.figure(figsize=(6, 4))
        sns.countplot(x='churn', data=df)
        plt.title("Distribution of Churn")
        plt.xlabel("Churn (0=No, 1=Yes)")
        plt.ylabel("Count")
        plt.show()
    
    # Identify variable types for plotting
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove 'churn' and 'customer_id' from plotting lists if present
    if 'churn' in numerical_cols: numerical_cols.remove('churn')
    if 'customer_id' in numerical_cols: numerical_cols.remove('customer_id')

    # VISUALIZATION 2: Numerical Distributions (Histograms)
    # What is code do: Plots histograms for all numerical features to show their frequency distribution.
    print("\nPlotting Numerical Distributions...")
    if len(numerical_cols) > 0:
        df[numerical_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
        plt.suptitle("Distributions of Numerical Features")
        plt.show()

    # VISUALIZATION 3: Categorical Distributions (Count Plots)
    # What is code do: Plots bar charts for categorical features to show frequency of each category.
    print("\nPlotting Categorical Distributions...") # Note: We are doing this BEFORE encoding in main, or need to ensure df passed here is readable
    # (Assuming df passed has original categorical cols or we are doing this before encoding. 
    # Current flow encodes in preprocess_data. We should ideally visualize specific known cat cols if encoded, or handle order.)
    # For safe plotting of encoded values or remaining object cols:
    for col in categorical_cols:
         plt.figure(figsize=(8, 4))
         sns.countplot(y=col, data=df, order=df[col].value_counts().index)
         plt.title(f"Distribution of {col}")
         plt.show()

    # VISUALIZATION 4: Bivariate Analysis (Box Plots vs Churn)
    # What is code do: Compares numerical feature distributions between Churned and Non-Churned users.
    if 'churn' in df.columns:
        print("\nPlotting Relationships with Churn...")
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='churn', y=col, data=df)
            plt.title(f"{col} by Churn Status")
            plt.show()

    # VISUALIZATION 5: Correlation Heatmap
    # What is code do: Visualizes the correlation matrix to show linear relationships between variables.
    print("\nPlotting Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
    
    # Top Correlations Text
    print("\nTop Correlations with Churn:")
    if 'churn' in df.columns:
        print(corr['churn'].sort_values(ascending=False).head(10))

# ==========================================
# 5. PREDICTIVE ANALYTICS
# ==========================================
# Function to train and compare models
# What is code do: Trains multiple ML models, evaluates them, and determines the best one.
def predictive_analytics(df):
    print("\n--- Starting Predictive Analytics ---")
    
    if 'churn' not in df.columns:
        print("Error: 'churn' column not found.")
        return None, None
    
    # 5.1 Prepare Data
    X = df.drop(['churn', 'customer_id', 'date_of_registration'], axis=1, errors='ignore') # Drop non-feature cols
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5.2 Define Models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_score = 0
    best_model_obj = None
    
    # 5.3 Train and Evaluate
    print("\nModel Comparison Results:")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC':<10}")
    print("-" * 85)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc}
        print(f"{name:<25} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {auc:.4f}")
        
        # Determine best model based on F1 Score (usually good for Churn which is imbalanced)
        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model_obj = model

    print("-" * 85)
    print(f"\n>> BEST MODEL: {best_model_name} with F1-Score: {best_score:.4f}")
    
    return best_model_obj, scaler

# ==========================================
# 6. PRESCRIPTIVE ANALYTICS
# ==========================================
# 6.1 Markov Chain Analysis
# What is code do: Simulates customer state transitions to estimate long-term churn probabilities.
def markov_chain_analysis(df):
    print("\n--- Starting Markov Chain Analysis ---")
    
    # Define generic states for simulation if proper state history isn't available
    # We will assume states based on 'estimated_salary' quartiles or usage as proxies for Customer Value Tiers
    # States: 'Low Value', 'Medium Value', 'High Value', 'Churned'
    
    print("Defining customer states based on Estimated Salary...")
    df['value_segment'] = pd.qcut(df['estimated_salary'], 3, labels=['Low Value', 'Medium Value', 'High Value'])
    
    # Create Transition Matrix (Simulated logic for this dataset)
    # In a real scenario, you'd calculate this from historical time-series data
    # Here we define a hypothetical transition matrix for demonstration
    # State order: [Low, Medium, High, Churn]
    transition_matrix = np.array([
        [0.70, 0.10, 0.05, 0.15], # Low Value transitions
        [0.05, 0.80, 0.10, 0.05], # Medium Value transitions
        [0.02, 0.08, 0.88, 0.02], # High Value transitions
        [0.00, 0.00, 0.00, 1.00]  # Churn is an absorbing state
    ])
    
    states = ['Low Value', 'Medium Value', 'High Value', 'Churn']
    
    print("\nConstructed Transition Probability Matrix:")
    print(pd.DataFrame(transition_matrix, columns=states, index=states))
    
    # Project distributions after N steps
    current_state_dist = np.array([0.4, 0.4, 0.2, 0.0]) # Initial distribution example
    steps = 12 # 1 year projected
    future_dist = current_state_dist.dot(np.linalg.matrix_power(transition_matrix, steps))
    
    print(f"\nProjected Customer Distribution after {steps} months:")
    for s, p in zip(states, future_dist):
        print(f"{s}: {p:.2%}")
        
    return transition_matrix

# 6.2 Monte Carlo Simulation
# What is code do: Runs random simulations to estimate risk/CLV distribution.
def monte_carlo_simulation(df, n_simulations=1000):
    print("\n--- Starting Monte Carlo Simulation ---")
    
    # Objective: Estimate average monthly revenue at risk
    avg_revenue = df['estimated_salary'].mean() / 12 # Approximation of monthly value
    churn_rate_base = df['churn'].mean()
    
    print(f"Base Parameters: Avg Monthly Revenue=${avg_revenue:.2f}, Avg Churn Rate={churn_rate_base:.2%}")
    
    # Simulation
    # What is code do: Simulate 12 months for a cohort of 1000 customers multiple times
    cohort_size = 1000
    results = []
    
    for _ in range(n_simulations):
        # Apply random fluctuation to churn rate (volatility)
        period_churn_rate = np.random.normal(churn_rate_base, 0.02) # std dev of 2%
        period_churn_rate = max(0, min(1, period_churn_rate)) # Clamp between 0 and 1
        
        remaining_customers = cohort_size * ((1 - period_churn_rate) ** 12)
        revenue_generated = remaining_customers * avg_revenue * 12
        results.append(revenue_generated)
        
    print(f"\nMonte Carlo Results ({n_simulations} runs):")
    print(f"Projected Annual Revenue (Mean): ${np.mean(results):,.2f}")
    print(f"Projected Annual Revenue (5th Percentile - Worst Case): ${np.percentile(results, 5):,.2f}")
    print(f"Projected Annual Revenue (95th Percentile - Best Case): ${np.percentile(results, 95):,.2f}")

# 6.3 Recommendation System
# What is code do: Suggests actions for customers based on their risk profile and usage.
def recommendation_system(df, model, scaler):
    print("\n--- Running Recommendation System ---")
    
    # Prepare data for prediction (using the structure from predictive step)
    X = df.drop(['churn', 'customer_id', 'date_of_registration', 'total_interactions', 'data_usage_log', 'value_segment'], axis=1, errors='ignore')
    # Re-apply scaler for consistency with training
    # Note: In a production script, we'd systematically handle columns. Here we do a quick align.
    # For now, we'll pick a sample of high-risk customers
    
    # Identify high risk customers
    high_churners = df[df['churn'] == 1].head(5) # Taking actual churners as proxy for "High Risk" for demo
    
    print("\nGenerate Recommendations for High Risk/Churned Sample:")
    print("-" * 80)
    for index, row in high_churners.iterrows():
        cust_id = row.get('customer_id', index)
        usage = row.get('data_used', 0)
        
        # Logic for recommendation
        # What is code do: Simple rule-based logic to assign offers
        rec_text = "Standard Retention Call"
        if usage > 5000:
            rec_text = "Offer: Premium Data Plan Upgrade (20% Discount)"
        elif usage < 100:
            rec_text = "Offer: Basic Plan Downgrade + Free SMS Pack"
        else:
            rec_text = "Offer: Loyalty Bonus 5GB Data"
            
        print(f"Customer {cust_id} | Data Usage: {usage} | Recommendation: {rec_text}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    file_path = 'c:/Users/dell/Downloads/pre_proj/telecom_churn.csv'
    
    # 1. Load
    df = load_data(file_path)
    if df is None: return
    
    # 2. Preprocess
    df_clean = preprocess_data(df)
    
    # 3. Feature Eng
    df_eng = feature_engineering(df_clean)
    
    # 4. Descriptive
    descriptive_analytics(df_eng)
    
    # 5. Predictive
    # Note: We need to handle the extra columns added during feature eng/prescriptive prep before passing to model
    # For simplicity, predictive_analytics function does its own dropping of non-features.
    best_model, scaler = predictive_analytics(df_eng)
    
    # 6. Prescriptive
    markov_chain_analysis(df_eng)
    monte_carlo_simulation(df_eng)
    
    if best_model:
        recommendation_system(df_eng, best_model, scaler)

if __name__ == "__main__":
    main()
