#!/usr/bin/env python
# coding: utf-8

# LaTex Markdown and Generating Data with Numpy

# # Notebook Imports and Styles

# In[166]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Example
# 
# $$f(x) = x^2 + X +1$$
# $f(x) = x^2 + X + 1$

# # Batch Gradient Descent

# In[167]:


def f(x):
    return x**2 + x + 1


# In[168]:


def df(x):
    return 2*x + 1


# In[169]:


# Make Data
x_1 = np.linspace (start=-3, stop=3, num=500)


# In[170]:


plt.figure(figsize=[15,5])

# Plot
plt.subplot(1, 2, 1)
plt.xlim([-3,3])
plt.ylim([0,8])
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.grid()
plt.ylabel('f(x)')
plt.plot(x_1, f(x_1), color='r', linewidth=3)


# 2 Chart:Derivative 
plt.subplot(1, 2, 2) #(row, column, index)
plt.xlabel('x', fontsize=16)
plt.ylabel('df(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 3)
plt.ylim(-3, 6)
plt.plot(x_1, df(x_1), color='g', linewidth=5)

plt.show


# ## Slope & Derivatives
# Challenge: Create a python function for the derivative of $f(x)$ called $df(x)$

# In[171]:


def df(x):
    return 2*x + 1


# ## Python Loops & Gradient Descent

# In[172]:


#Python For Loop
for n in range(5):
    print('Hello World', n)
print ('End of loop')


# In[173]:


#Python While Loop
counter = 2
while counter < 7:
    print('Counting ...', counter)
    counter = counter + 1
print ('Ready or not, here I come!')


# In[174]:


#Gradient Descent
new_x = 3
previous_x = 0
step_multiplier = .1
precision = .00001

x_list = []
slope_list = []


for n in range (40):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - step_multiplier * gradient
    
    x_list.append(new_x)
    slope_list.append(df(new_x))
    
    step_size = abs(new_x-previous_x)
    if step_size <precision:
        print('Loop ran this many times:', n)
        break 

print('Local minimum occurs at:', new_x)
print('Slope or df(x) value at this point is:', df(new_x))
print('f(x) value or cost at this point is:', f(new_x))


# In[175]:


plt.figure(figsize=[15,5])

# Plot
plt.subplot(1, 3, 1)
plt.xlim([-3,3])
plt.ylim([0,8])
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.grid()
plt.ylabel('f(x)')
plt.plot(x_1, f(x_1), color='r', linewidth=3, alpha=.7)

values = np.array(x_list)
plt.scatter(x_list, f(values), color='g', s=100, alpha=.6)


# 2 Chart:Derivative 
plt.subplot(1, 3, 2) #(row, column, index)
plt.xlabel('x', fontsize=16)
plt.ylabel('df(x)', fontsize=16)
plt.grid()
plt.xlim(-2, 3)
plt.ylim(-3, 6)
plt.plot(x_1, df(x_1), color='g', linewidth=5, alpha=.7)

plt.scatter(x_list, slope_list, color='r', s=100, alpha=.6)

#3 Chart: Derivative (Close Up)
plt.subplot (1,3,3)
plt.title('Gradient Descent (close up)', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.grid()
plt.xlim(-.55, -.2)
plt.ylim(-.3, .8)

plt.plot(x_1, df(x_1), color='g', linewidth=5, alpha=.7)
plt.scatter(x_list, slope_list, color='r', s=300, alpha=.6)


plt.show


# ### Example 2 - Multiple Minima vs Initial Guess & Advanced Functions
# #### $$g(x) = x^4 - 4x^2 + 5$$

# In[176]:


# Make some data
x_2 = np.linspace(-2, 2, 1000)

# Challenge: Write the g(x) function and the dg(x) function in Python?
def g(x):
   return x**4 - 4*x**2 + 5

def dg(x):
   return 4*x**3 -8*x


# In[177]:


plt.figure(figsize=[15,5])

#1 chart: Cost Function
plt.subplot(1,2,1)
plt.xlim(-2,2)
plt.ylim(0.5,5.5)

plt.title('Cost Function', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), color='blue', linewidth=3)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()

plt.xlim(-2,2)
plt.ylim(-6,8)

plt.plot(x_2, dg(x_2), color='skyblue', linewidth=5)


# ## Gradient Descent as a Python Function

# In[178]:


#Inputs: derivative function, initial guess, multiplier/learning-rate, precision
def gradient_descent(derivative_func, initial_guess, multiplier=.02, precision=.001, max_iter=100):
    
#want to indent multiple lines together press: ctrl + ]

    newer_x = initial_guess

    x_list = [newer_x]
    slope_list=[derivative_func(newer_x)]

    for n in range (max_iter):
        previous_x = newer_x
        gradient = derivative_func(previous_x)
        newer_x = previous_x - multiplier*gradient

        step_size = abs(newer_x - previous_x)

        x_list.append(newer_x)
        slope_list.append(derivative_func(newer_x))

        if step_size < precision:
            break
    return newer_x, x_list, slope_list


# In[179]:


local_min, list_x, deriv_list = gradient_descent(dg,.05, .02, .001)
print ('Local min occurs at:', local_min)
print ('Number of steps:', len(list_x))


# In[180]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=-.5, 
                                                 multiplier=.02, precision=.001)
print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[181]:


local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=-.1)
print('Local min occurs at:', local_min)
print('Number of steps:', len(list_x))


# In[191]:


#Calling gradient descent function
local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess= 0.1)

#Plot function and derivative and scatter plot side to side

plt.figure(figsize=[15,5])

#1 chart: Cost Function
plt.subplot(1,2,1)
plt.xlim(-2,2)
plt.ylim(0.5,5.5)

plt.title('Cost Function', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), color='blue', linewidth=3, alpha=.8)
plt.scatter(list_x, g(np.array(list_x)), color='red', s=100, alpha=.6

#2 Chart: Derivative
plt.subplot(1,2,2)
            
plt.title ('Slope of the cost function', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()

plt.xlim(-2,2)
plt.ylim(-6,8)

plt.plot(x_2, dg(x_2), color='skyblue', linewidth=5, a)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=.5)


# ## Example 3: Divergence, Overflow and Python Tuple
# ### $$h(x) = x^5 -2x^4 +2$$

# In[ ]:


#Make Data
x_3 =np.linspace(start = -2.5, stop=2.5, num=1000)

def h(x):
    return x**5 - 2*x**4 +2

def dh(x):
    return 5*x**4 - 8*x**3 


# In[184]:


#Calling gradient descent function
local_min, list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess=-0.2, max_iter=70)

#Plot function and derivative and scatter plot side by side
plt.figure(figsize=[15,5])

#1 Chart: Cost function
plt.subplot (1,2,1)

plt.xlim(-1.2,2.5)
plt.ylim(-1, 4)

plt.title('Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('h(x)', fontsize=16)

plt.plot(x_3, h(x_3), color='blue', linewidth=3, alpha=0.8)
plt.scatter(list_x, h(np.array(list_x)), color='red', s=100, alpha=.6)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dh(x)', fontsize=16)
plt.grid()
plt.xlim(-1,2)
plt.ylim(-4,5)

plt.plot(x_3, dh(x_3), color='skyblue', linewidth=5, alpha=.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=.5)

plt.show()


# In[ ]:


import sys
sys.float_info.max


# In[183]:


#Creating a tuple -tuple packing
breakfast = 'bacon', 'eggs', 'avocado'
unlucky_numbers = 13, 4, 9, 26, 17

#How to access a value in a tuple
print('I loooove ', breakfast[0])
print('My hotel has no ' + str(unlucky_numbers[1]) + 'th floor')

not_my_address= 1, 'Infinite Loop', 'Cupertino', 95014

tuple_with_single_value = 42,
type(tuple_with_single_value)

main, side, greens = breakfast
print('Main course is ', main)

data_tuple = gradient_descent(derivative_func=dh, initial_guess=.2)
print('Local min is', data_tuple[0])
print('Cost at the last x value is', h(data_tuple[0]))
print('Number of steps is', len(data_tuple[1]))


# ## The Learning Rate

# In[202]:


#Calling gradient descent function
local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=1.9, 
                                                 multiplier=.02, max_iter=500)

#Plot function and derivative and scatter plot side by side
plt.figure(figsize=[15,5])

#1 Chart: Cost function
plt.subplot (1,2,1)

plt.xlim(-2,2)
plt.ylim(0.5, 5.5)

plt.title('Cost Function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('g(x)', fontsize=16)

plt.plot(x_2, g(x_2), color='blue', linewidth=3, alpha=0.8)
plt.scatter(list_x, g(np.array(list_x)), color='red', s=100, alpha=.6)

#2 Chart: Derivative
plt.subplot(1,2,2)

plt.title('Slope of the cost function', fontsize=17)
plt.xlabel('X', fontsize=16)
plt.ylabel('dg(x)', fontsize=16)
plt.grid()
plt.xlim(-2,2)
plt.ylim(-6,8)

plt.plot(x_2, dg(x_2), color='skyblue', linewidth=5, alpha=.6)
plt.scatter(list_x, deriv_list, color='red', s=100, alpha=.5)

plt.show()

print('Number of steps is:', len(list_x))


# In[224]:


#Run gradient descent 3 times
n=100
low_gamma = gradient_descent(derivative_func=dg, initial_guess=3,
                            multiplier=.0005, precision=.0001, max_iter=n)

mid_gamma = gradient_descent(derivative_func=dg, initial_guess=3,
                            multiplier=.001, precision=.0001, max_iter=n)

high_gamma = gradient_descent(derivative_func=dg, initial_guess=3,
                             multiplier=.002, precision=.0001, max_iter=n)

insane_gamma = gradient_descent(derivative_func=dg, initial_guess=1.9,
                              multiplier=.25, precision=.0001, max_iter=n)

#Plotting reduction in cost for each iteration
plt.figure(figsize=[20,10])

plt.xlim(0,n)
plt.ylim(0,50)

plt.title('Effect of the learning rate', fontsize=17)
plt.xlabel('Number of iterations', fontsize=16)
plt.ylabel('Cost', fontsize=16)

#Values for our charts
#1) Y axis Data: convert the lists to numpy arrays
low_values = np.array(low_gamma[1])
mid_values = np.array(mid_gamma[1])
high_values = np.array(high_gamma[1])
insane_values = np.array(insane_gamma[1])

#2 X Axis Data: create a list from 0 to n+1
iteration_list= list(range(0, n+1))

#Plotting Low Learning Rate
plt.plot(iteration_list, g(low_values), color='blue', linewidth=3, alpha=.8)
plt.scatter(iteration_list, g(low_values), color='blue', s=80, alpha=.6)

#Plotting Mid Learning Rate
plt.plot(iteration_list, g(mid_values), color='green', linewidth=3, alpha=.8)
plt.scatter(iteration_list, g(mid_values), color='green', s=80, alpha=.6)

#Plotting High Learning Rate
plt.plot(iteration_list, g(high_values), color='purple', linewidth=3, alpha=.8)
plt.scatter(iteration_list, g(high_values), color='purple', s=80, alpha=.6)

#Plotting Insane Learning Rate
plt.plot(iteration_list, g(insane_values), color='red', linewidth=3, alpha=.8)
plt.scatter(iteration_list, g(insane_values), color='red', s=80, alpha=.6)


# # Stochastic Gradient Descent

# In[ ]:




