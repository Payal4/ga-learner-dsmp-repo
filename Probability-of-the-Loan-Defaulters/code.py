# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here

df = pd.read_csv(path)
total = len(df)

p_a = len(df[df.fico > 700]) / total
p_b = len(df[df.purpose == 'debt_consolidation']) / total
print(p_a)
print(p_b)
df1 = df[df.purpose == 'debt_consolidation']
p_a_b = len(df1[df1.fico > 700]) / len(df[df.fico > 700])
print(p_a_b)
p_b_a = p_a_b * p_a / p_b
print(p_b_a)
result = p_b_a == p_a
#if p_b_a == p_a:
#    result = "Independent"
#else:
#    result = "Not Independent"

print(result)
# code ends here


# --------------
# code starts here

prob_lp = len(df[df['paid.back.loan'] == 'Yes']) / len(df)
prob_cs = len(df[df['credit.policy'] == 'Yes']) / len(df)

new_df = df[df['paid.back.loan'] == 'Yes']

prob_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes']) / len(new_df) 
 
bayes = (prob_pd_cs * prob_lp) / prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
df1 = df[df['paid.back.loan'] == 'No']
purpose = df1['purpose'].value_counts(ascending=False)
purpose.plot(kind='bar')
plt.xlabel('Purpose')
plt.ylabel('No of Loan Defaulters')
plt.title('Probability of the Loan Defaulters')
plt.show()

# code ends here


# --------------
# code starts here

inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
inst_mean

plt.hist(df['installment'], bins=10)
plt.axvline(x=inst_median, color='green')
plt.axvline(x=inst_mean, color='black')
plt.show()

plt.hist(df['log.annual.inc'], bins=10)

# code ends here


