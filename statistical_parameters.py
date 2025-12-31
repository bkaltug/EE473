import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os 
import os

# Current Workspace
# print(os.getcwd()) Data file must be here 
df = pd.read_csv('dataset/heart_cleveland_upload.csv')
print(df)

# Calculate counts
counts = df['condition'].value_counts()

# Calculate percentages (normalize=True gives the proportion)
percentages = df['condition'].value_counts(normalize=True) * 100

# Combine into a nice table
summary_table = pd.DataFrame({
    'Count': counts, 
    'Percentage (%)': percentages.round(2)
})

print(summary_table) # We have a balanced dataset


# 1. Define your lists manually based on your knowledge of the data
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope','ca','thal','condition']

# 2. Analyze Numerical Data (Mean, Median, Std Dev)
print("--- Numerical Statistics ---")
num_stats = df[numerical_cols].describe().T[['mean', '50%', 'std', 'min', 'max']]
num_stats = num_stats.rename(columns={'50%': 'median'})
num_stats['skewness'] = df[numerical_cols].skew()
num_stats['kurtosis'] = df[numerical_cols].kurt()
num_stats = num_stats.round(2)
print(num_stats)

# 3. Analyze Categorical Data (We want Counts and Proportions)
print("\n--- Categorical Statistics ---")
for col in categorical_cols:
    # Calculate the breakdown of categories
    counts = df[col].value_counts()
    percent = df[col].value_counts(normalize=True) * 100
    
    # Combine into a readable dataframe
    summary = pd.DataFrame({'Count': counts, 'Percentage (%)': percent.round(1)})
    
    print(f"\nVariable: {col}")
    print(summary)


def corrMat(df, id=False):
    # Calculate correlation
    corr_mat = df.corr().round(2)
    
    # Setup plot
    f, ax = plt.subplots(figsize=(10, 5))
    
    # Create mask (using 'bool' instead of 'np.bool')
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    
    # Shift the matrix (remove first row and last col) to remove the diagonal
    mask = mask[1:, :-1]
    corr = corr_mat.iloc[1:, :-1].copy()
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, vmin=-0.3, vmax=0.3, center=0, 
                cmap='RdPu_r', square=False, lw=2, annot=True, cbar=False)
    
    ax.set_title('Shifted Linear Correlation Matrix')
    plt.show() # Make sure to show the plot!

# Call the function OUTSIDE the definition (no indentation)
corrMat(df)

''' CountPlot Histograms '''

plt4 = ['#E379B2','#6351BB']
def plot1count(x,xlabel,palt):
    
    plt.figure(figsize=(20,2))
    sns.countplot(x=x,hue='condition', data=df, palette=palt)
    plt.legend(["Heart Disease Negative", "Heart Disease "],loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()
    
def plot1count_ordered(x,xlabel,order,palt):
    
    plt.figure(figsize=(20,2))
    sns.countplot(x=x,hue='condition',data=df,order=order,palette=palt)
    plt.legend(["Heart Disease Negative", "Heart Disease Positive"],loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.show()

def plot2count(x1,x2,xlabel1,xlabel2,colour,rat,ind1=None,ind2=None):
    
    # colour, ratio, index_sort

    fig,ax = plt.subplots(1,2,figsize=(20,3),gridspec_kw={'width_ratios':rat})
    # Number of major vessels (0-3) colored by flourosopy
    sns.countplot(x=x1,hue='condition',data=df,order=ind1,palette=colour,ax=ax[0])
    ax[0].legend(["Heart Disease Negative", "Heart Disease Positive"],loc='upper right')
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel('Frequency')

    # Defect Information (0 = normal; 1 = fixed defect; 2 = reversable defect )
    sns.countplot(x=x2,hue='condition', data=df,order=ind2,palette=colour,ax=ax[1])
    ax[1].legend(["Heart Disease Negative", "Heart Disease Positive"],loc='best')
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel('Frequency')
    plt.show()
    
''' Plot n Countplots side by side '''
def nplot2count(lst_name,lst_label,colour,n_plots):
    
    ii=-1;fig,ax = plt.subplots(1,n_plots,figsize=(20,3))
    for i in range(0,n_plots):
        ii+=1;id1=lst_name[ii];id2=lst_label[ii]
        sns.countplot(x=id1,hue='condition',data=df,palette=colour,ax=ax[ii])
        ax[ii].legend(["Heart Disease Negative", "Heart Disease Positive"],loc='upper right')
        ax[ii].set_xlabel(id2)
        ax[ii].set_ylabel('Frequency')

plot2count('age','sex','Age of Patient','Gender of Patient',plt4,[2,1])
lst1 = ['cp','exang','thal','ca']
lst2 = ['Chest Pain Type','Excersised Induced Angina','Thalium Stress Result','Fluorosopy Vessels']
nplot2count(lst1,lst2,plt4,4)

lst_ecg = ['oldpeak','restecg','slope','condition']
plot1count('oldpeak','oldpeak: ST Depression Relative to Rest',plt4)
plot2count('restecg','slope','restecg: Resting electrocardiography (ECG)','slope: []',plt4,[1,1])

lst_blood = ['trestbps','thalach','fbs','chol','condition']
plot1count('trestbps','trestbps: Resting Blood Pressure (mmHg)',plt4)
plot1count_ordered('thalach','thalach: Maximum Heart Rate',df['thalach'].value_counts().iloc[:30].index,plt4)
plot2count('fbs','chol','Fasting Blood Sugar','Serum Cholestoral',plt4,[2,10],None,df['chol'].value_counts().iloc[:40].index)