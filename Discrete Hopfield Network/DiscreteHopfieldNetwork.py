"""
Title: Discrete Hopfield Network
Code by: Niranjan Ketkar

"""

import numpy as np
import itertools

#Given input vectors are
s1 = np.array([[1, 1, -1, -1, -1, 1]])
s2 = np.array([[1, -1, -1, 1, -1, -1]])
s3 = np.array([[-1, -1, 1, 1, 1, -1]])
s4 = np.array([[-1, 1, 1, -1, 1, 1]])

s = np.array([s1, s2, s3, s4])

#initialize weight vector with zeros
w = np.zeros((s[0][0].shape[0], s[0][0].shape[0]))

#calculate weight vector
for si in s:
    w+=(si.T.dot(si))

#set diagonal elements to zero, no self connections
np.fill_diagonal(w, 0)

#The weight vector is:
print('The weight vector is:')
print(w)
"""
output:
[[ 0.  0. -4.  0. -4.  0.]
 [ 0.  0.  0. -4.  0.  4.]
 [-4.  0.  0.  0.  4.  0.]
 [ 0. -4.  0.  0.  0. -4.]
 [-4.  0.  4.  0.  0.  0.]
 [ 0.  4.  0. -4.  0.  0.]]
"""
#Function to get equilibrium state of the given input vector s_in
def get_equi_state(s_in, win):
    w = np.array(win)
    count = 0
    #storing previous  s_in to check equilibrium state
    s_in_old = np.array(s_in)
    #boolean array to check if we got convergence for all the values of y
    is_converged = np.zeros(s_in[0].shape)
    
    #we stop when for all the values of neuron index we converge
    while(not is_converged.all()):
        ni = int(np.random.randint(len(w), size = 1))
        temp = s_in[0][ni] + w[ni].dot(s_in[0])
        if temp != 0:
            s_in[0][ni] = int(temp>0)*2-1
        else:
            s_in[0][ni] = s_in_old[0][ni]
        #checking if s_in is changed or not
        if (s_in == s_in_old).all():
            is_converged[ni] = 1
        else:
            #if changed, then reset all ticked values and start counting again
            is_converged = np.zeros(is_converged.shape)
        s_in_old = np.array(s_in)
    #return the equilibrium state
    return s_in

#1. Checking  whether all the stored patterns are equilibrium states or not
#If its an equilibrium state, then it will converge to itself when it is given
#as an input
s1_out = get_equi_state(s1, w)
if (s1_out == s1).all():
    print("Vector s1 is an equilibrium state")
else:
    print("Vector s1 is not an equilibrium state")
s2_out = get_equi_state(s2, w)
if (s2_out == s2).all():
    print("Vector s2 is an equilibrium state")
else:
    print("Vector s2 is not an equilibrium state")
s3_out = get_equi_state(s3, w)
if (s3_out == s3).all():
    print("Vector s3 is an equilibrium state")
else:
    print("Vector s3 is not an equilibrium state")
s4_out = get_equi_state(s4, w)
if (s4_out == s4).all():
    print("Vector s4 is an equilibrium state")
else:
    print("Vector s4 is not an equilibrium state")
"""
output:
Vector s1 is an equilibrium state
Vector s2 is an equilibrium state
Vector s3 is an equilibrium state
Vector s4 is an equilibrium state
"""

#2. Lets find all the equilibrium states of the system
#we will find all unique converged states
input_vec_list = []
converged_state_list = []
pattern_dict = {}
stored_pattern_dict = {}
#generate all possible combinations of 1 and -1 and 0 of same size of store vectors
#which is nothing but binary conversion 0 to 63
#or we can use built in library function itertools
vec_iter = itertools.product([-1, 1], repeat = len(w))

#Try all possible combinations as input pattern
for y_in in vec_iter:
    input_vec_list.append(y_in)
    y_input = np.array(y_in).reshape(1, -1)
    equi_state = get_equi_state(y_input, w)
    converged_state_list.append(tuple(equi_state[0]))

#we will find unique patterns among converged list
converged_state_set = set(tuple(converged_state_list))
print("The possible equilibrium states are:")
for converged_state in converged_state_set:
    print(converged_state)
    pattern_dict[converged_state] = []

"""
output:
The possible equilibrium states are:
(1, -1, -1, 1, -1, -1)
(-1, -1, 1, 1, 1, -1)
(1, 1, -1, -1, -1, 1)
(-1, 1, 1, -1, 1, 1)

"""
for si in s:
    stored_pattern_dict[tuple(si[0])] = []

diff = set(stored_pattern_dict.keys()).difference(set(pattern_dict.keys()))
if not diff:
    print("There are no spurious states for the network")
else:
    print("The spurious states of the network are:", diff)
"""
output:
There are no spurious states for the network
"""
#3. Now lets find the basin of attraction for each equilibrium state
for pattern_counter in range(len(input_vec_list)):
    pattern_dict[converged_state_list[pattern_counter]].append(input_vec_list[pattern_counter])

#Lets print the basin of attraction for each key
for key in pattern_dict.keys():
    print("The basin of attraction for pattern" + str(key) +"is:", pattern_dict[key])
"""
output:
('The basin of attraction for pattern(1, -1, -1, 1, -1, -1)is:', [(-1, -1, -1, -1, -1, -1), (-1, -1, -1, 1, -1, -1), (-1, -1, -1, 1, -1, 1), (-1, 1, -1, 1, -1, -1), (1, -1, -1, -1, -1, -1), (1, -1, -1, -1, 1, -1), (1, -1, -1, 1, -1, -1), (1, -1, -1, 1, -1, 1), (1, -1, -1, 1, 1, -1), (1, -1, -1, 1, 1, 1), (1, -1, 1, -1, -1, -1), (1, -1, 1, 1, -1, -1), (1, -1, 1, 1, -1, 1), (1, 1, -1, 1, -1, -1), (1, 1, -1, 1, 1, -1), (1, 1, 1, 1, -1, -1)])
('The basin of attraction for pattern(-1, 1, 1, -1, 1, 1)is:', [(-1, -1, -1, -1, 1, 1), (-1, -1, 1, -1, -1, 1), (-1, -1, 1, -1, 1, 1), (-1, 1, -1, -1, 1, -1), (-1, 1, -1, -1, 1, 1), (-1, 1, -1, 1, 1, 1), (-1, 1, 1, -1, -1, -1), (-1, 1, 1, -1, -1, 1), (-1, 1, 1, -1, 1, -1), (-1, 1, 1, -1, 1, 1), (-1, 1, 1, 1, -1, 1), (-1, 1, 1, 1, 1, 1), (1, -1, 1, -1, 1, 1), (1, 1, 1, -1, 1, -1), (1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 1)])
('The basin of attraction for pattern(-1, -1, 1, 1, 1, -1)is:', [(-1, -1, -1, -1, 1, -1), (-1, -1, -1, 1, 1, -1), (-1, -1, -1, 1, 1, 1), (-1, -1, 1, -1, -1, -1), (-1, -1, 1, -1, 1, -1), (-1, -1, 1, 1, -1, -1), (-1, -1, 1, 1, -1, 1), (-1, -1, 1, 1, 1, -1), (-1, -1, 1, 1, 1, 1), (-1, 1, -1, 1, 1, -1), (-1, 1, 1, 1, -1, -1), (-1, 1, 1, 1, 1, -1), (1, -1, 1, -1, 1, -1), (1, -1, 1, 1, 1, -1), (1, -1, 1, 1, 1, 1), (1, 1, 1, 1, 1, -1)])
('The basin of attraction for pattern(1, 1, -1, -1, -1, 1)is:', [(-1, -1, -1, -1, -1, 1), (-1, 1, -1, -1, -1, -1), (-1, 1, -1, -1, -1, 1), (-1, 1, -1, 1, -1, 1), (1, -1, -1, -1, -1, 1), (1, -1, -1, -1, 1, 1), (1, -1, 1, -1, -1, 1), (1, 1, -1, -1, -1, -1), (1, 1, -1, -1, -1, 1), (1, 1, -1, -1, 1, -1), (1, 1, -1, -1, 1, 1), (1, 1, -1, 1, -1, 1), (1, 1, -1, 1, 1, 1), (1, 1, 1, -1, -1, -1), (1, 1, 1, -1, -1, 1), (1, 1, 1, 1, -1, 1)])
"""
#4. Lets find out the probability that given input pattern does not associate with
#the stored pattern

#sum of counts of all patterns which are converging to stored pattern
match_count = 0
for s_vec in s:
    s_vec = s_vec.reshape(1, -1)
    match_count+=(len(pattern_dict[tuple(s_vec[0])]))
unmatched_count = len(input_vec_list) - match_count
print("probability that given pattern does not associate with stored pattern is = ",    \
        str(unmatched_count)+"/"+str(len(input_vec_list))+ " = ",
        float(unmatched_count)/float(len(input_vec_list)))
"""
output:
('probability that given pattern does not associate with stored pattern is = ', '0/64 = ', 0.0)
"""
"""
Code to find huffman distance for each vector with its converged state
"""
for key in pattern_dict.keys():
    for pattern in pattern_dict[key]:
        print(6 - sum((np.array(key) == np.array(pattern)).astype(np.int)))
    print("end of basin")
"""
output:
2
1
2
2
1
2
0
1
1
2
2
1
2
1
2
2
end of basin
2
2
1
2
1
2
2
1
1
0
2
1
2
2
1
2
end of basin
2
1
2
2
1
1
2
0
1
2
2
1
2
1
2
2
end of basin
2
2
1
2
1
2
2
1
0
2
1
1
2
2
1
2

"""
