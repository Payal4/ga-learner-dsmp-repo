# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  

# path        [File location variable]

#Code starts here

data = pd.read_csv(path)
data_sample = data.sample(n=sample_size, random_state= 0)
sample_mean = data_sample.installment.mean()
print("Sample mean : ", sample_mean)
sample_std = data_sample.installment.std()
print("Sample Standard Deviation : ", sample_std)
margin_of_error = z_critical *(sample_std / math.sqrt(sample_size))
print("Margin of error : ", margin_of_error)
confidence_interval = (sample_mean - margin_of_error) , (sample_mean + margin_of_error)
print("Confidence Interval : ", confidence_interval)
true_mean = data.installment.mean()
print("True mean : ", true_mean)
check = ((true_mean>= confidence_interval[0]) and (true_mean <= confidence_interval[1]))

print("True mean falls in the range of Confidence Interval : ", check)





# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here

fig, axes = plt.subplots(nrows = 3, ncols = 1)
for i in range(len(sample_size)):
    m=[]
    for j in range(1000):
        data_sample = data['installment'].sample(n=sample_size[i])
        m.append(data_sample.mean())
    mean_series = pd.Series(m)
    axes[i].hist(mean_series)





# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here

data["int.rate"] = data['int.rate'].str.replace('%', '').astype(float)
data["int.rate"] = data["int.rate"] / 100
data["int.rate"]

x1 = data[data['purpose']=='small_business']['int.rate']
value = data['int.rate'].mean()
alternative = ''
z_statistic , p_value = ztest(x1=x1, value=value, alternative='larger')
print("Z Statistic : ",z_statistic)
print("P-value : ",p_value)
less = p_value< 0.05
greater = p_value> 0.05

if less==True:
    print("The Null Hypothesis is Rejected")
elif greater==True:
    print("The Null Hypothesis is not Rejected")





# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here

x1 = data[data['paid.back.loan']=='No']['installment']
x2 = data[data['paid.back.loan']=='Yes']['installment']
z_statistic, p_value = ztest(x1=x1, x2=x2)
print("Z Statistic : ",z_statistic)
print("P-value : ",p_value)
less = p_value< 0.05
greater = p_value> 0.05

if less==True:
    print("The Null Hypothesis is Rejected")
elif greater==True:
    print("The Null Hypothesis is not Rejected")





# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here

yes = data[data["paid.back.loan"] == "Yes"]["purpose"].value_counts()
no = data[data["paid.back.loan"] == "No"]["purpose"].value_counts()

observed = pd.concat([yes.transpose(),no.transpose()], axis=1, keys=['Yes','No'])
print(observed)
chi2, p, dof, ex = stats.chi2_contingency(observed)

print("Chi-square statistic = ",chi2)
print("p-value = ",p)

check = chi2 > critical_value

if check == True:
    print("The Alternate Hypothesis is Accepted")
else :
    print("The Null Hypothesis is Accepted")






