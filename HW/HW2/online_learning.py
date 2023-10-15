# Importing standard libraries
import numpy as np
from math import factorial

### Define function 
def cal_one (main_string):
    sub_string = '1'
    start_index=0
    count_er=0

    for i in range(len(main_string)):
      j = main_string.find(sub_string,start_index)
      if(j!=-1):
        start_index = j+1
        count_er+=1
    return count_er

def cal_comb(N,x):
    return factorial(N) / (factorial(x)*factorial(N-x))

### Main below
# Reading lines of binary outcomes from a file
with open("testfile.txt", "r") as file:
    data = file.read().split()
size = len(data)

# Getting initial parameters a and b from the user
a = int(input("Enter Initial Parameters:\na: "))
b = int(input("b: "))

for i in range(size):
    print(f"case {i+1}: {data[i]}")
    # Calculating Likelihood

    N = len(data[i])
    x = cal_one (data[i]) # times of one showing up
    factorial_N_x = cal_comb(N,x)
    #p = 0.5
    #(p can be inferred with Beta(a,b) the peak value of the distribution)
    #(watch out the special case (p=1) when Beta(10, 1))
    p = (a - 1) / (a + b - 2) # peak value of beta function: (a-1)/(a+b-2)
    print(N, x, factorial_N_x)
    case_likelihood = factorial_N_x*p**x*(1-p)**(N-x)
    print(f"Likelihood: {case_likelihood}" )

    # Calculating Prior
    prior = (a - 1) / (a + b - 2)  # peak value of beta function: (a-1)/(a+b-2)
    print(f"Beta prior: a = {a} b = {b}")

    # Calculating Posterior
    posterior = case_likelihood * prior
    #print(f"Posterior: {posterior}")

    # Calculating new parameters 'a' and 'b'
    a = a + x  # a = a + k; where k is the number of 1's in one iteration
    b = b + len(data[i]) - x  # b = b + n - k; where k is the number of 1's in one iteration
    print(f"Beta Posterior:a: {a} b: {b}\n")