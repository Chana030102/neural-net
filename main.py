# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 2
#

import numpy as np 
import pandas, pickle
import network, sys, os

np.seterr(all='ignore')
NAME      = input("Enter name of experiment: ")

if (len(sys.argv) != 1):
    if sys.argv[1] == '-c':
        hidden_layer_size = int(input("Enter size of hidden layer: "))
        momentum = int(input("Enter NN momentum: "))
        learning_rate = int(input("Enter NN learning rate: "))
    else:
        print("{} is not a valid option. You can use \"-c\" to customize the NN run.".format(sys.argv[1]))
        sys.exit(0)

else:
    hidden_layer_size = 100
    momentum = 0.9
    learning_rate = 0.1

PATH = "{}_h={}_m={}_lr={}/".format(NAME,hidden_layer_size,momentum,learning_rate)

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
TITLE_TRAIN= 'train'
TITLE_TEST = 'test'
DELIMITER  = ','

output_layer_size = 10
MAX_EPOCH = 1 
INPUT_MAX = 255
increment = 1000

# Make directory if it doesn't exist
print("All generated files will be stored in {}".format(PATH))
if os.path.exists(PATH) == False:
    os.mkdir(PATH)
    print("Directory created")

# Import data
print("Importing Data...")
traind = np.loadtxt(TRAIN_FILE,delimiter=DELIMITER)
testd  = np.loadtxt(TEST_FILE,delimiter=DELIMITER)

# Pre-process data
# Shuffle, separate labels, and scale
print("Processing data for NN...")
np.random.shuffle(traind)
trainl = traind[:,0]
traind = np.delete(traind,0,axis=1)
traind = np.divide(traind, INPUT_MAX)

np.random.shuffle(testd)
testl  = testd[:,0]
testd  = np.delete(testd,0,axis=1)
testd  = np.divide(testd,INPUT_MAX)

t1 = int(len(traind)/increment)
t2 = int(len(testd)/increment)

print("Generating pickle dump files for data...")
for i in range(t1):
    pickle.dump(traind[(i*increment):((i+1)*increment)],open(PATH+TITLE_TRAIN+str(i),'wb'))

for i in range(t2):
    pickle.dump(testd[(i*increment):((i+1)*increment)],open(PATH+TITLE_TEST+str(i),'wb'))

input_size = len(testd[0]) # how many inputs are in one row
del testd,traind

print("Creating NN and starting experiment...")
net = network.NeuralNet(input_size,hidden_layer_size,output_layer_size,momentum,learning_rate)

# Observe inital epoch 0 accuracy and train for 50 epochs
# Observe accuracy after each epoch
for e in range(MAX_EPOCH):
    
    for i in range(t1):
        print("Train file: {}".format(i))
        traind = pickle.load(open(PATH+TITLE_TRAIN+str(i),'rb'))
        net.evaluate(TITLE_TRAIN,traind,trainl[i*increment:(i+1)*increment])
        del traind
    
    for i in range(t2):
        print("Test file: {}".format(i))
        testd = pickle.load(open(PATH+TITLE_TEST+str(i),'rb'))
        net.evaluate(TITLE_TEST,testd,testl[i*increment:(i+1)*increment])
        del testd
    
    for i in range(t1):
        print("Train file: {}".format(i))
        traind = pickle.load(open(PATH+TITLE_TRAIN+str(i),'rb'))
        net.train(traind,trainl[i*increment:(i+1)*increment])
        del traind

    net.epoch += 1

# Observe 50th epoch accuracy results
# Create confusion matrix for test data testing
for i in range(t1):
    print("Train file: {}".format(i))
    traind = pickle.load(open(PATH+TITLE_TRAIN+str(i),'rb'))
    net.evaluate(TITLE_TRAIN,traind,trainl[i*increment:(i+1)*increment])
    del traind

print("Running {} and generating confusion matrix".format(net.epoch))
for i in range(t2):
    print("Test file: {}".format(i))
    testd = pickle.load(open(PATH+TITLE_TEST+str(i),'rb'))
    net.evaluate(TITLE_TEST,testd,testl[i*increment:(i+1)*increment],True)
    net.train(testd,testl[i*increment:(i+1)*increment])
    del testd

print("Generating accuracy and confusion matrix files")
net.report_accuracy(PATH+NAME)
net.report_confusion_matrix(PATH+NAME)
