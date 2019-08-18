# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data = np.genfromtxt(path, delimiter=',', skip_header=1)

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
#Code starts here
census=np.concatenate((data,new_record))
print(census)


# --------------
#Code starts here
age=np.array(census[:,:1])
print(age)
max_age=age.max()
min_age=age.min()
age_mean=age.mean()
age_std=np.std(age)
print("The maximum age is ",max_age)
print("The minimum age is " ,min_age)
print("The mean of the age is " ,age_mean)
print("The standard deviation of the age is " ,age_std)


# --------------
#Code starts here
race_0=census[census[:,2]==0]
race_1=census[census[:,2]==1]
race_2=census[census[:,2]==2]
race_3=census[census[:,2]==3]
race_4=census[census[:,2]==4]

len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)

minimum_race=min(len_0,len_1,len_2,len_3,len_4)

minority_race=0
if len_0 == minimum_race:
    minority_race = 0
elif len_1 == minimum_race:
    minority_race = 1
elif len_2 == minimum_race:
    minority_race = 2
elif len_3 == minimum_race:
    minority_race = 3
elif len_4 == minimum_race:
    minority_race = 4

print('The race with minimum no of citizens is ',minority_race)






# --------------
#Code starts here
senior_citizens = census[census[:,0] > 60]
sum_array = np.sum(senior_citizens, axis=0)
working_hours_sum=sum_array[6]
senior_citizens_len = len(senior_citizens)
avg_working_hours = working_hours_sum/senior_citizens_len

print("The average working hours of senior citizens is : ",avg_working_hours)

if avg_working_hours>25:
    print("The govt. policy is not followed")
else:
    print("the govt. policy is followed")


# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
avg_pay_high = (sum(high[:,7])) / len(high)
avg_pay_low = (sum(low[:,7])) / len(low)
print('The average pay of higher educated people is : ', avg_pay_high)
print('The average pay of less educated people is : ',avg_pay_low)

if avg_pay_high>avg_pay_low:
    print("The average salary of higher educated people is  \ngreater than less educated people")
else:
    print("The average salary of less educated people is \ngreater than higher educated people")



