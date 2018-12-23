import numpy as np
from itertools import combinations

#function to convert given input 
# into bipolar form
def makeBipolarTrainingSet(training):
    for i in range(len(training)):
        if training[i] == '.':
            training[i] = -1
        else:
            training[i] = 1
    return training

#function to print bipolar
#vector as original input
def printVector(vector):
    for i in range(5):
        for j in range(3):
            vector_num = vector[3*i+j]
            if(vector_num==-1):
                print '.',
            elif(vector_num == 1):
                print '#',
            else:
                #print ! for unknown state
                print '!',
        print(" ")

#Use Hebb's learning rule for NN
#to predict whether given input is C
def isC(w, b, training):
    mult_output = np.matmul(w, training) + b
    if mult_output == 0:
        return 0
    elif mult_output>0:
        return 1
    else:
        return -1

# C, which means output is 1
training1 = ".###..#..#...##"
t1 = 1
# not a C so t2= -1
training2 = "#.##.#.#.#.##.#"
t2 = -1

training1 = np.array(makeBipolarTrainingSet(list(training1.strip()))).reshape((-1, 1))
training2 = np.array(makeBipolarTrainingSet(list(training2.strip()))).reshape((-1, 1))

w = training1.T*t1 + training2.T*t2
b = t1+t2

print "weight w is:", w

#output:
#weight w is: [-2  2  0  0  0 -2  2 -2  0  0  0 -2 -2  2  0]

#Lets consider the scenario when C has mistake in one of its pixels
#lets flip each pixel of C one by one and see what we get as output
one_pixel_flipped_output = []
for i in range(len(training1)):
    training1[i]*=-1
    one_pixel_flipped_output.append(isC(w, b, training1))
    training1[i]*=-1

print "predicted output for all combinations:", one_pixel_flipped_output
#output:
#[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#So flipping one pixel, does not cause problem for NN in detecting C.

#function to try all combinations by changing pixel values
def try_all_combinations(training_vector, flipping_factor=-1, flipping_count_start=2, flipping_count_end=16):
    combinations_count_vector = []
    flipped_pixel_unknown_output = []
    flipped_pixel_misclassified_vector = []
    first_misclassified_vector= []
    first_unknown_output_vector = []
    found_first_misclassified = False
    found_first_unknown_output = False
    ukout_count_pixels_flipped = -1
    misclassified_count_pixels_flipped = -1
    
    #go on flipping 2, 3...all pixels
    for flipped_pixel_count in range(flipping_count_start, flipping_count_end):    
        all_combs = combinations(range(15), flipped_pixel_count)
        flipped_pixel_output = []
    
        #trying different combinations for changing pixels    
        for combination in all_combs:
            temp_training = training_vector.copy()
            for index in combination:
                #making new vector with changed pixels
                temp_training[index]*=flipping_factor
        
            temp_output = isC(w, b, temp_training)
            if temp_output == -1 and not found_first_misclassified:
                first_misclassified_vector = temp_training
                found_first_misclassified = True
                misclassified_count_pixels_flipped = flipped_pixel_count
            if temp_output == 0 and not found_first_unknown_output:
                first_unknown_output_vector = temp_training
                found_first_unknown_output = True
                ukout_count_pixels_flipped = flipped_pixel_count
                
            flipped_pixel_output.append(temp_output)
        
        flipped_pixel_misclassified_vector.append(flipped_pixel_output.count(-1))
        combinations_count_vector.append(len(flipped_pixel_output))
        flipped_pixel_unknown_output.append(flipped_pixel_output.count(0))
    return combinations_count_vector, flipped_pixel_unknown_output, flipped_pixel_misclassified_vector, first_misclassified_vector, misclassified_count_pixels_flipped, first_unknown_output_vector, ukout_count_pixels_flipped

#now we will flip 2 or more pixels of C with all possible comibinations and then check
#whether NN is still recognizing it as C or not

combinations_count_vector = [len(one_pixel_flipped_output)]
flipped_pixel_unknown_output = [one_pixel_flipped_output.count(0)]
flipped_pixel_misclassified_vector = [one_pixel_flipped_output.count(-1)]

comb, un, miscl, first_misclassified_vector, m_pcount, first_unknown_output_vector, u_pcount = try_all_combinations(training1, -1)

combinations_count_vector += comb
flipped_pixel_unknown_output += un
flipped_pixel_misclassified_vector += miscl
misclassified_num_pixels_flipped = m_pcount
ukoutput_num_pixels_flipped = u_pcount

print "Vector Representing total no of combinations tried for pixel count = 1 to pixel count 15:"
print combinations_count_vector
#output:
#[15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1]

print "Vector representing total number of times C was misclassified(output is 'not a C' when k pixels were flipped:)"
print flipped_pixel_misclassified_vector
#output:
#[0, 0, 0, 0, 56, 420, 1380, 2605, 3115, 2457, 1295, 455, 105, 15, 1]

print "Vector representing total number of times output was given as undetermined state when k pixels were flipped:"
print flipped_pixel_unknown_output
#output:[0, 0, 0, 70, 490, 1470, 2450, 2450, 1470, 490, 70, 0, 0, 0, 0]

print "Vector representing total no of times we didn't predicted C:"
print np.add(flipped_pixel_unknown_output, flipped_pixel_misclassified_vector)
#output:
#[   0    0    0   70  546 1890 3830 5055 4585 2947 1365  455  105   15    1]

print "we got our first vector which is misclassified after flipping", misclassified_num_pixels_flipped, "pixels"
#output:5

print "The first vector after flipping pixels which is misclassified as 'not a C' is:\n"
printVector(np.squeeze(first_misclassified_vector))
'''
output:

# . #  
# . #  
. # .  
# . .  
. # #

'''
print "we got our first output of 'unknown output' or output as zero after flipping", ukoutput_num_pixels_flipped, "pixels"
#output:4

print "The first vector for which we got unknown output is:\n"
printVector(np.squeeze(first_unknown_output_vector))
'''
output:

# . #  
# . #  
. . .  
# . .  
. # #  

'''


print "---------------------------------------------------------------------------------"

#Now we will try varous combinations by setting few of the pixels as undetermined and then 
#will check whether NN is still recognizing it as C or not

comb, un, miscl, first_misclassified_vector, m_pcount, first_unknown_output_vector, u_pcount = try_all_combinations(training1, 0, 1, 16)

combinations_count_vector = comb
und_pixel_unknown_output_vector = un
un_pixel_misclassified_vector = miscl
misclassified_num_pixels_und = m_pcount
ukoutput_num_pixels_und = u_pcount

print "Vector Representing total no of combinations tried for pixel count = 1 to pixel count 15:"
print combinations_count_vector
#output:
#[15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1]

print "Vector representing total number of times C was misclassified(output is 'not a C' when k pixels were set as undetermined state:)"
print un_pixel_misclassified_vector
#output:
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print "Vector representing total number of times output was given as undetermined state when k pixels were set as undetermined state:"
print und_pixel_unknown_output_vector
#output:
#[0, 0, 0, 0, 0, 0, 0, 1, 7, 21, 35, 35, 21, 7, 1]

print "Vector representing total no of times we didn't predicted C:"
print np.add(und_pixel_unknown_output_vector, un_pixel_misclassified_vector)
#output:
#[ 0  0  0  0  0  0  0  1  7 21 35 35 21  7  1]

if misclassified_num_pixels_und != -1:
    print "we got our first vector which is misclassified after setting", misclassified_num_pixels_und, "pixels as undetermined"
    print first_misclassified_vector
    print "The first vector after setting undetermined pixels which is misclassified as 'not a C' is:\n", printVector(first_misclassified_vector)
else:
    print "we didnt got any misclassified C after setting all combinations of pixels as undetermined(Thus we either got undetermined output or we predicted a C)"

print "we got our first output of 'unknown output' or output as zero after setting", ukoutput_num_pixels_und, "pixels undetermined"
#output:8

print "The first vector for which we got unknown output is:\n"
printVector(np.squeeze(first_unknown_output_vector))
'''
output:

! ! #  
# . !  
! ! .  
# . !  
! ! # 


'''
