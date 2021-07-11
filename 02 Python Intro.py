#!/usr/bin/env python
# coding: utf-8

# In[1]:


myAge = 32
print(myAge)


# In[2]:


myAge = 33
print(myAge)


# In[3]:


print(myAge/3)


# In[4]:


myAge = myAge + 1
print(myAge)


# In[5]:


# Challenge
# 1) Create a variable called 'restaurantBill' and set its value to 36.17
restaurantBill = 36.17
# 2) Create a variable called 'serviceCharge' and set its value to .125
serviceCharge = .125
# 3) Print out the amount of tip
tip = restaurantBill * serviceCharge
print(tip)


# In[6]:


type(33)


# In[7]:


type(33.6)


# In[8]:


type('Syed')


# In[9]:


type(myAge)


# In[10]:


type(restaurantBill)


# Lists and Arrays

# In[11]:


primeNumbers = [3,7,61,29,199]
type(primeNumbers)


# In[12]:


coolPeople = ['Jay Z', 'Martin Luther King', 'Ghandi']
primeAndPeople = ['King Arthur', 17,11, 'Jennifer Lopez']
type(primeAndPeople)


# In[13]:


primeNumbers[2]


# In[14]:


bestPrimeEver = primeNumbers[4]
print(bestPrimeEver)


# In[15]:


randomList = [6, 'cat', 'hello', 23]
print (randomList)


# In[16]:


randomArray = [5,6,3,5]
print (randomArray)


# In[17]:


type(randomArray)


# In[18]:


import pandas as pd
data = pd.read_csv('lsd_math_score_data.csv')


# In[19]:


print(data)


# In[20]:


type(data)


# Python DataFrames and Series

# In[21]:


import pandas as pd
data = pd.read_csv('lsd_math_score_data.csv')


# In[22]:


print(data)
type(data)


# In[23]:


data['Avg_Math_Test_Score']


# In[24]:


MathScores = data['Avg_Math_Test_Score']
print (MathScores)


# In[25]:


data['Test_Subject'] = 'Jennifer Lopez'


# In[26]:


print(data)


# In[27]:


data['High_Score'] = 100
print(data)


# In[28]:


data['High_Score'] = MathScores + 100


# In[29]:


data['High_Score']


# In[30]:


data['High_Score'] = data['High_Score'] + data['Avg_Math_Test_Score']
print(data)


# In[31]:


#d_HighScore = data['High_Score'] 
#d_HighScore = d_HighScore**2
#print(d_HighScore)


# In[32]:


data['High_Score'] = data['High_Score']**2
print(data)


# In[33]:


type(data['High_Score'])


# In[34]:


#columnList = [data['LSD_ppm'], data['Avg_Math_Test_Score']]
#print(columnList)


# In[35]:


#columnList = ['LSD_ppm', 'Avg_Math_Test_Score']
#cleanData = data[columnList]
cleanData = data[['LSD_ppm', 'Avg_Math_Test_Score']]
print(cleanData)


# In[36]:


Y = data[['Avg_Math_Test_Score']]
type(Y)


# In[37]:


Y = data['Avg_Math_Test_Score']
type(Y)


# In[38]:


X = data[['LSD_ppm']]
print(X)
type(X)


# In[39]:


del data['Test_Subject']
print(data)


# In[40]:


del data['High_Score']
print(data)


# Python Function

# Part 1: Defining and Calling Functions

# In[41]:


def get_theWin():
    print('get sword')
    print('slay the dragon')
    print('GET THE GOLD')


# In[42]:


get_theWin()


# Part 2: Arguments & Parameters

# In[43]:


def milk_mission(amount, destination):
    print('Open Door')
    print('Walk to the' + destination)
    print('Buy'+ amount + 'cartona on the ground floor')
    print('Return with milk galore')


# In[44]:


milk_mission(' twenty ', ' department')


# In[45]:


milk_mission(amount=' twenty ', destination= ' department')


# Part 3: Results & Return Values

# In[46]:


def times(firstInput, secondInput):
    answer = firstInput * secondInput
    return answer


# In[47]:


times(3,5)


# In[48]:


import this


# Objects: Understanding Attributes and Methods

# In[49]:


import life as hitchhikersGuide


# In[50]:


hitchhikersGuide.quote_marvin()


# In[51]:


result = hitchhikersGuide.square_root(9)
print(result)


# How to Make Sense of Pythin Documentation for Data Visualisation

# In[52]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[53]:


time = data['Time_Delay_in_Minutes']
LSD = data['LSD_ppm']
score = data['Avg_Math_Test_Score']


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.title('Tissue concentration of LSD over time', fontsize=17)
plt.xlabel('Time in Minutes', fontsize = 14)
plt.ylabel('Tissue LSD ppm', fontsize = 14)
plt.text(x=0, y=-.5, s='Wagner et al. (1968)', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(1,7)
plt.xlim(0,500)

plt.style.use('classic')

plt.plot(time, LSD, color= 'c', linewidth=3)
plt.show()


# Working with Python Objects to Analyse Data

# In[88]:


time = data[['Time_Delay_in_Minutes']]
LSD = data[['LSD_ppm']]
score = data[['Avg_Math_Test_Score']]

regr = LinearRegression()
regr.fit(LSD, score)
predicted_score = regr.predict(LSD)
print('Theta 1:', regr.coef_[0][0])
print('Intercept: ', regr.intercept_[0])
print('R-Square: ', regr.score(LSD, score))


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.title('Arithmetic vs LSD-25', fontsize=17)
plt.xlabel('Tissue LSD ppm', fontsize=14)
plt.ylabel('Performance Score', fontsize=14)
plt.ylim(25, 85)
plt.xlim(1, 6.5)
plt.style.use('fivethirtyeight')

plt.scatter(LSD, score, color = 'blue', s = 300, alpha = .3)
plt.plot(LSD, predicted_score, color='red', linewidth=3)
plt.show()


# In[ ]:




