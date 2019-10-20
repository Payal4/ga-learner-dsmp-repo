# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
# code starts here

bank = pd.read_csv(path)

categorical_var = bank.select_dtypes(include='object')               
print(categorical_var)
numerical_var = bank.select_dtypes(include='number')
print(numerical_var)
# code ends here


# --------------
# code starts here

banks = bank.drop(['Loan_ID'],axis=1)
#print(banks)
print(banks.isnull().sum())
bank_mode = banks.mode()

banks.fillna(value = {'Gender':bank_mode['Gender'].iloc[0] ,
'Married':bank_mode['Married'].iloc[0] ,
'Dependents':bank_mode['Dependents'].iloc[0] ,
'Education':bank_mode['Education'].iloc[0] ,
'Self_Employed':bank_mode['Self_Employed'].iloc[0] ,
'ApplicantIncome':bank_mode['ApplicantIncome'].iloc[0] ,
'CoapplicantIncome':bank_mode['CoapplicantIncome'].iloc[0] ,
'LoanAmount':bank_mode['LoanAmount'].iloc[0] ,
'Loan_Amount_Term':bank_mode['Loan_Amount_Term'].iloc[0] ,
'Credit_History':bank_mode['Credit_History'].iloc[0] ,
'Property_Area':bank_mode['Property_Area'].iloc[0] ,
'Loan_Status':bank_mode['Loan_Status'].iloc[0] }, inplace=True)


print(banks.isnull().sum())
print(banks)
#code ends here




# --------------
# Code starts here




avg_loan_amount = pd.pivot_table(banks, index=['Gender','Married','Self_Employed'], values=['LoanAmount'])

print(avg_loan_amount)
# code ends here



# --------------
# code starts here




loan_approved_se = len(banks[(banks['Self_Employed']=="Yes") & (banks['Loan_Status']                         =="Y")])
print(loan_approved_se)

loan_approved_nse = len(banks[(banks['Self_Employed']=="No") & (banks['Loan_Status']                         =="Y")])
print(loan_approved_nse)

loan_status_count = len(banks['Loan_Status'])

percentage_se = loan_approved_se/loan_status_count * 100
print('Percentage of loan approval for self employed people: '+ str(percentage_se) + '%')

percentage_nse = loan_approved_nse/loan_status_count * 100
print('Percentage of loan approval for people who are not self-employed: '+ str(percentage_nse) + '%')



# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x:x/12)
#print(loan_term)
big_loan_term = len(loan_term[loan_term>=25])
print("The number of applicants having loan amount term greater than or equal to 25 years: " + str(big_loan_term))
# code ends here


# --------------
# code starts here




loan_groupby = banks.groupby('Loan_Status')[['ApplicantIncome','Credit_History']]
print(loan_groupby)
mean_values = loan_groupby.mean()
# code ends here


