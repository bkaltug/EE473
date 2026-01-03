import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/heart_cleveland_upload.csv")

# Split groups
thalach_disease = df[df["condition"] == 1]["thalach"]
thalach_healthy = df[df["condition"] == 0]["thalach"]


s1 = thalach_disease.std(ddof=1)
s2 = thalach_healthy.std(ddof=1)

var1 = s1**2
var2 = s2**2

F = var1 / var2 if var1 > var2 else var2 / var1

df1 = len(thalach_disease) - 1
df2 = len(thalach_healthy) - 1

p_var = 2 * min(
    stats.f.cdf(F, df1, df2),
    1 - stats.f.cdf(F, df1, df2)
)

print("VARIANCE TEST (F-test)")
print("F statistic:", F)
print("p-value:", p_var)


t_stat, p_two_tail = stats.ttest_ind(
    thalach_disease,
    thalach_healthy,
    equal_var=False
)

p_one_tail = p_two_tail / 2 if t_stat < 0 else 1 - (p_two_tail / 2)

print("\nT-TEST (Welch, one-tailed)")
print("t statistic:", t_stat)
print("one-tailed p-value:", p_one_tail)


plt.figure(figsize=(12,5))

# Histogram + KDE
plt.subplot(1,2,1)
sns.histplot(thalach_healthy, color="green", kde=True, stat="density", label="Healthy", alpha=0.5)
sns.histplot(thalach_disease, color="red", kde=True, stat="density", label="Disease", alpha=0.5)
plt.xlabel("Thalach")
plt.title("Distribution of Thalach")
plt.legend()

# Boxplot
plt.subplot(1,2,2)
sns.boxplot(x="condition", y="thalach", data=df)
plt.xticks([0,1], ["Healthy", "Disease"])
plt.title("Thalach Comparison")

plt.tight_layout()
plt.show()