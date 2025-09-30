# Comprehensive Exploratory Data Analysis (EDA) Framework
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import normaltest, jarque_bera, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# 1. MEANINGFUL QUESTIONS ABOUT THE DATASET
# ============================================================================

def ask_meaningful_questions():
    """
    Define meaningful questions to guide the EDA process
    """
    questions = {
        "Data Quality": [
            "Are there any missing values in the dataset?",
            "Are there any duplicate records?",
            "Are data types appropriate for each variable?",
            "Are there any inconsistent data formats?",
            "Are there any unrealistic or impossible values?"
        ],
        "Data Distribution": [
            "What is the distribution of each numerical variable?",
            "Are the numerical variables normally distributed?",
            "What is the frequency distribution of categorical variables?",
            "Are there any class imbalances in categorical variables?"
        ],
        "Relationships": [
            "What are the correlations between numerical variables?",
            "Are there any strong linear or non-linear relationships?",
            "How do categorical variables relate to numerical outcomes?",
            "Are there any interaction effects between variables?"
        ],
        "Patterns & Anomalies": [
            "Are there any seasonal or time-based patterns?",
            "What outliers exist and are they meaningful?",
            "Are there any unexpected patterns in the data?",
            "Do certain groups behave differently than others?"
        ]
    }
    
    print("=== MEANINGFUL QUESTIONS TO GUIDE EDA ===\n")
    for category, q_list in questions.items():
        print(f"{category.upper()}:")
        for i, question in enumerate(q_list, 1):
            print(f"  {i}. {question}")
        print()
    
    return questions

# ============================================================================
# 2. COMPREHENSIVE DATA STRUCTURE EXPLORATION
# ============================================================================

def explore_data_structure(df):
    """
    Comprehensive exploration of data structure and basic information
    """
    print("=== DATA STRUCTURE EXPLORATION ===\n")
    
    # Basic information
    print("1. DATASET OVERVIEW:")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Display first and last few rows
    print("2. SAMPLE DATA:")
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print()
    
    # Data types and info
    print("3. DATA TYPES AND INFO:")
    print(df.info())
    print()
    
    # Detailed data type analysis
    print("4. DETAILED DATA TYPE ANALYSIS:")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        print(f"   {dtype}: {len(cols)} columns - {cols}")
    print()
    
    # Missing values analysis
    print("5. MISSING VALUES ANALYSIS:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("   No missing values found!")
    print()
    
    # Duplicate values
    print("6. DUPLICATE ANALYSIS:")
    duplicate_count = df.duplicated().sum()
    print(f"   Total duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        print("   Sample duplicate rows:")
        print(df[df.duplicated()].head())
    print()
    
    # Unique values for each column
    print("7. UNIQUE VALUES ANALYSIS:")
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_percent = (unique_count / len(df)) * 100
        print(f"   {col}: {unique_count} unique values ({unique_percent:.1f}%)")
    print()

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================

def comprehensive_descriptive_stats(df):
    """
    Generate comprehensive descriptive statistics
    """
    print("=== DESCRIPTIVE STATISTICS ===\n")
    
    # Numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print("1. NUMERICAL VARIABLES:")
        print(df[numerical_cols].describe())
        print()
        
        # Additional statistics
        print("2. ADDITIONAL NUMERICAL STATISTICS:")
        additional_stats = pd.DataFrame({
            'Variance': df[numerical_cols].var(),
            'Std Dev': df[numerical_cols].std(),
            'Skewness': df[numerical_cols].skew(),
            'Kurtosis': df[numerical_cols].kurtosis(),
            'Range': df[numerical_cols].max() - df[numerical_cols].min()
        })
        print(additional_stats)
        print()
    
    # Categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print("3. CATEGORICAL VARIABLES:")
        for col in categorical_cols:
            print(f"\n{col.upper()}:")
            value_counts = df[col].value_counts()
            value_percentages = df[col].value_counts(normalize=True) * 100
            summary = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages
            })
            print(summary)

# ============================================================================
# 4. ADVANCED VISUALIZATION FOR PATTERNS AND TRENDS
# ============================================================================

def visualize_patterns_and_trends(df):
    """
    Create comprehensive visualizations to identify patterns and trends
    """
    print("=== PATTERNS AND TRENDS VISUALIZATION ===\n")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Distribution plots for numerical variables
    if numerical_cols:
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Histogram with KDE
            sns.histplot(df[col], kde=True, stat='density')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            
            # Add mean and median lines
            mean_val = df[col].mean()
            median_val = df[col].median()
            plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # 2. Box plots for outlier detection
    if numerical_cols:
        plt.figure(figsize=(15, 6))
        df[numerical_cols].boxplot(figsize=(15, 6))
        plt.title('Box Plots for Outlier Detection')
        plt.xticks(rotation=45)
        plt.show()
    
    # 3. Correlation analysis
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix of Numerical Variables')
        plt.show()
        
        # Print strong correlations
        print("STRONG CORRELATIONS (|correlation| > 0.7):")
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((correlation_matrix.columns[i], 
                                     correlation_matrix.columns[j], corr_val))
        
        if strong_corr:
            for var1, var2, corr in strong_corr:
                print(f"   {var1} ↔ {var2}: {corr:.3f}")
        else:
            print("   No strong correlations found.")
        print()
    
    # 4. Categorical variable visualization
    if categorical_cols:
        n_cols = min(2, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 5*n_rows))
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            
            value_counts = df[col].value_counts()
            
            if len(value_counts) <= 10:  # Bar plot for few categories
                sns.countplot(data=df, x=col)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
            else:  # Show only top 10 categories
                top_10 = value_counts.head(10)
                sns.barplot(x=top_10.values, y=top_10.index)
                plt.title(f'Top 10 Categories in {col}')
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# 5. STATISTICAL HYPOTHESIS TESTING
# ============================================================================

def perform_hypothesis_testing(df):
    """
    Perform various statistical tests to validate assumptions
    """
    print("=== HYPOTHESIS TESTING & STATISTICAL VALIDATION ===\n")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Normality tests
    if numerical_cols:
        print("1. NORMALITY TESTS:")
        print("Testing if numerical variables follow normal distribution...")
        
        for col in numerical_cols:
            # Shapiro-Wilk test (for small samples)
            if len(df[col].dropna()) <= 5000:
                stat, p_value = stats.shapiro(df[col].dropna())
                test_name = "Shapiro-Wilk"
            else:
                # Jarque-Bera test (for larger samples)
                stat, p_value = jarque_bera(df[col].dropna())
                test_name = "Jarque-Bera"
            
            is_normal = "YES" if p_value > 0.05 else "NO"
            print(f"   {col}: {test_name} p-value = {p_value:.4f} | Normal: {is_normal}")
        print()
    
    # 2. Independence tests for categorical variables
    if len(categorical_cols) >= 2:
        print("2. INDEPENDENCE TESTS (Chi-square):")
        print("Testing independence between categorical variables...")
        
        for i in range(len(categorical_cols)):
            for j in range(i+1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                
                # Create contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])
                
                # Chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                is_independent = "YES" if p_value > 0.05 else "NO"
                print(f"   {col1} vs {col2}: χ² = {chi2:.4f}, p-value = {p_value:.4f} | Independent: {is_independent}")
        print()
    
    # 3. Comparison tests between groups
    if numerical_cols and categorical_cols:
        print("3. GROUP COMPARISON TESTS:")
        
        for num_col in numerical_cols:
            for cat_col in categorical_cols:
                unique_categories = df[cat_col].nunique()
                
                if unique_categories == 2:  # Two groups - use t-test
                    groups = df[cat_col].unique()
                    group1_data = df[df[cat_col] == groups[0]][num_col].dropna()
                    group2_data = df[df[cat_col] == groups[1]][num_col].dropna()
                    
                    if len(group1_data) > 0 and len(group2_data) > 0:
                        stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        significant = "YES" if p_value < 0.05 else "NO"
                        print(f"   {num_col} by {cat_col}: t-test p-value = {p_value:.4f} | Significant: {significant}")
                
                elif 2 < unique_categories <= 5:  # Multiple groups - use ANOVA
                    groups_data = [df[df[cat_col] == cat][num_col].dropna() for cat in df[cat_col].unique()]
                    groups_data = [group for group in groups_data if len(group) > 0]
                    
                    if len(groups_data) > 1:
                        stat, p_value = stats.f_oneway(*groups_data)
                        significant = "YES" if p_value < 0.05 else "NO"
                        print(f"   {num_col} by {cat_col}: ANOVA p-value = {p_value:.4f} | Significant: {significant}")
        print()

# ============================================================================
# 6. COMPREHENSIVE DATA ISSUES DETECTION
# ============================================================================

def detect_data_issues(df):
    """
    Comprehensive detection of data quality issues
    """
    print("=== DATA ISSUES DETECTION ===\n")
    
    issues_found = []
    
    # 1. Missing values pattern analysis
    print("1. MISSING VALUES PATTERN ANALYSIS:")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("   Missing values detected:")
        for col, count in missing_data[missing_data > 0].items():
            percentage = (count / len(df)) * 100
            print(f"     {col}: {count} ({percentage:.1f}%)")
            if percentage > 20:
                issues_found.append(f"High missing rate in {col}")
    else:
        print("   No missing values found.")
    print()
    
    # 2. Outlier detection using multiple methods
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print("2. OUTLIER DETECTION:")
        
        for col in numerical_cols:
            data = df[col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = len(data[(data < lower_bound) | (data > upper_bound)])
            
            # Z-score method (|z| > 3)
            z_scores = np.abs(stats.zscore(data))
            z_outliers = len(data[z_scores > 3])
            
            # Modified Z-score method
            median_val = np.median(data)
            mad = np.median(np.abs(data - median_val))
            modified_z_scores = 0.6745 * (data - median_val) / mad
            modified_z_outliers = len(data[np.abs(modified_z_scores) > 3.5])
            
            print(f"   {col}:")
            print(f"     IQR method: {iqr_outliers} outliers ({iqr_outliers/len(data)*100:.1f}%)")
            print(f"     Z-score method: {z_outliers} outliers ({z_outliers/len(data)*100:.1f}%)")
            print(f"     Modified Z-score: {modified_z_outliers} outliers ({modified_z_outliers/len(data)*100:.1f}%)")
            
            if iqr_outliers > len(data) * 0.1:  # More than 10% outliers
                issues_found.append(f"High outlier rate in {col}")
        print()
    
    # 3. Data consistency checks
    print("3. DATA CONSISTENCY CHECKS:")
    
    # Check for negative values where they shouldn't exist
    for col in numerical_cols:
        if any(keyword in col.lower() for keyword in ['age', 'price', 'amount', 'quantity', 'income']):
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"   WARNING: {negative_count} negative values in {col}")
                issues_found.append(f"Negative values in {col}")
    
    # Check for unrealistic ranges
    for col in numerical_cols:
        if 'age' in col.lower():
            unrealistic = ((df[col] < 0) | (df[col] > 120)).sum()
            if unrealistic > 0:
                print(f"   WARNING: {unrealistic} unrealistic age values in {col}")
                issues_found.append(f"Unrealistic age values in {col}")
    print()
    
    # 4. Duplicate analysis
    print("4. DUPLICATE ANALYSIS:")
    total_duplicates = df.duplicated().sum()
    print(f"   Total duplicate rows: {total_duplicates}")
    
    if total_duplicates > 0:
        duplicate_percentage = (total_duplicates / len(df)) * 100
        print(f"   Duplicate percentage: {duplicate_percentage:.2f}%")
        if duplicate_percentage > 5:
            issues_found.append(f"High duplicate rate ({duplicate_percentage:.1f}%)")
    print()
    
    # 5. Data type inconsistencies
    print("5. DATA TYPE INCONSISTENCIES:")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric values are stored as strings
            try:
                pd.to_numeric(df[col], errors='raise')
                print(f"   WARNING: {col} contains numeric data but stored as object")
                issues_found.append(f"Numeric data stored as text in {col}")
            except:
                pass
    print()
    
    # Summary of issues
    print("6. SUMMARY OF ISSUES FOUND:")
    if issues_found:
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("   No major data quality issues detected!")
    print()

# ============================================================================
# 7. MAIN EDA FUNCTION
# ============================================================================

def perform_comprehensive_eda(df):
    """
    Main function to perform comprehensive EDA
    """
    print("🔍 COMPREHENSIVE EXPLORATORY DATA ANALYSIS 🔍")
    print("="*60)
    
    # Step 1: Ask meaningful questions
    questions = ask_meaningful_questions()
    
    # Step 2: Explore data structure
    explore_data_structure(df)
    
    # Step 3: Descriptive statistics
    comprehensive_descriptive_stats(df)
    
    # Step 4: Visualize patterns and trends
    visualize_patterns_and_trends(df)
    
    # Step 5: Statistical testing
    perform_hypothesis_testing(df)
    
    # Step 6: Detect data issues
    detect_data_issues(df)
    
    print("="*60)
    print("✅ COMPREHENSIVE EDA COMPLETED!")
    print("="*60)

# ============================================================================
# 8. EXAMPLE USAGE WITH SAMPLE DATA
# ============================================================================

if _name_ == "_main_":
    # Create a more comprehensive sample dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data with various patterns and issues
    sample_data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples),
        'purchases': np.random.poisson(15, n_samples),
        'satisfaction_score': np.random.normal(4.2, 0.8, n_samples),
        'website_visits': np.random.exponential(5, n_samples).astype(int),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    # Create DataFrame
    df_sample = pd.DataFrame(sample_data)
    
    # Introduce some data quality issues for demonstration
    # Add missing values
    missing_indices = np.random.choice(df_sample.index, size=int(0.05 * n_samples), replace=False)
    df_sample.loc[missing_indices, 'satisfaction_score'] = np.nan
    
    # Add some duplicate rows
    duplicate_rows = df_sample.sample(n=20)
    df_sample = pd.concat([df_sample, duplicate_rows], ignore_index=True)
    
    # Add some outliers
    outlier_indices = np.random.choice(df_sample.index, size=30, replace=False)
    df_sample.loc[outlier_indices, 'income'] *= 10  # Extreme income values
    
    # Add some negative ages (data error)
    error_indices = np.random.choice(df_sample.index, size=5, replace=False)
    df_sample.loc[error_indices, 'age'] = -np.random.randint(1, 10, 5)
    
    print("Sample dataset created with intentional data quality issues for demonstration.")
    print("Dataset shape:", df_sample.shape)
    print("\n" + "="*60)
    
    # Perform comprehensive EDA
    perform_comprehensive_eda(df_sample)
    
    # ========================================================================
    # Instructions for using with your own dataset:
    # ========================================================================
    print("\n" + "="*60)
    print("📋 TO USE WITH YOUR OWN DATASET:")
    print("="*60)
    print("1. Replace the sample data creation section with:")
    print("   df = pd.read_csv('your_dataset.csv')")
    print("   # or")
    print("   df = pd.read_excel('your_dataset.xlsx')")
    print("   # or load from any other source")
    print()
    print("2. Then simply call:")
    print("   perform_comprehensive_eda(df)")
    print()
    print("3. The analysis will automatically adapt to your dataset structure!")
    print("="*60)
