import os, sys, re, random, time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

if len(sys.argv) != 2:
    print("Usage: python %s <data_folder>" %(sys.argv[0]))
    sys.exit(0)


sw = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
pos = [sys.argv[1]+"/txt_sentoken/pos/"+jk for jk in os.listdir(sys.argv[1]+"/txt_sentoken/pos/")]
neg = [sys.argv[1]+"/txt_sentoken/neg/"+jk for jk in  os.listdir(sys.argv[1]+"/txt_sentoken/neg/")]
pos.sort()
neg.sort()

#Function to create dataset
def build_dataset(pos_class, neg_class, gram=1):
    #Build bag of words usiong training data
    bow = []
    for fn in pos_class[:800]+neg_class[:800]:
        with open(fn, "r") as fh:
            fc = re.sub("[^\w']", " ", fh.read())
            if gram > 1:
                all_words = [x for x in re.sub("[\s]{2,}", " ", fc).split(" ") if x not in sw]
            else:
                all_words = re.sub("[\s]{2,}", " ", fc).split(" ")
            all_words = [" ".join([all_words[i+j] for j in range(gram)]) for i in range(len(all_words)-(gram-1))]

            for z in all_words:
                bow.append(z)
    all_words, fc = None, None


    bow = list(set(bow))
    #Building numpy array of data points
    x = np.zeros(shape=(2000, len(bow)))
    y = np.zeros(shape=(2000, 1))

    #Create dataset for training and testing
    doc_id = 0 
    for fn in pos+neg:
        doc_dict = {}
        with open(fn, "r") as fh:
            fc = re.sub("[^\w']", " ", fh.read())
            if gram > 1:
                all_words = [x for x in re.sub("[\s]{2,}", " ", fc).split(" ") if x not in sw]
            else:
                all_words = re.sub("[\s]{2,}", " ", fc).split(" ")
            all_words = [" ".join([all_words[i+j] for j in range(gram)]) for i in range(len(all_words)-(gram-1))]
            
            doc_words = Counter(all_words)
            for z in bow:
                doc_dict[z] = doc_words[z]

        x[doc_id] = list(doc_dict.values())
        if "pos" in fn: y[doc_id] = 1
        else: y[doc_id] = -1

        doc_id += 1
    
    doc_dict, all_words, fc = None, None, None #To free memory

    return x, y, bow

#Train function
def train(x, y, weights):
    all_weights = []
    #For each data point in training data
    for i in range(len(x)):
        #Perform prediction using dot product of weight and data point, Perform sign function for binary classification
        predict_y = 1 if np.dot(weights, x[i]) >= 0 else -1 
        
        #Update weight when prediction is wrong
        if y[i] != predict_y:
            if y[i] == 1:
                weights = weights + x[i]
            else:
                weights = weights - x[i]
        all_weights.append(weights)
    return all_weights


#Test function
def test(x, y, weights):
    accuracy = 0 
    for i in range(len(x)):
        predict_y = 1 if np.dot(weights,x[i]) >=0 else -1 

        if y[i] == predict_y:
            accuracy += 1
    #Compute and return accuracy of test
    return (accuracy/len(x))*100

def run_program(gram = 1):
    print("[*] Preparing dataset...")
    x, y, bow = build_dataset(pos, neg, gram)

    #Split data into testing and training
    print("[*] Splitting to training and testing data")
    pos_index = np.where(y == 1)[0].tolist() #Get all indices with positive sentiment in label
    neg_index = np.where(y == -1)[0].tolist() #Get all indices with negative sentiment in label

    all_pos_x = np.take(x, pos_index, axis=0) #Get all the positive sentiment data from the dataset
    all_neg_x = np.take(x, neg_index, axis=0) #Get all negative sentiment data from the dataset

    all_pos_y = np.take(y, pos_index, axis=0) #Extract all corresponding labels for positive sentiments
    all_neg_y = np.take(y, neg_index, axis=0) #Extract all corresponding labels for negative sentiments

    x, y = None, None #To free memory and prevent MemoryError
    
    #Split data into training and testing
    train_x, train_y = np.vstack((all_pos_x[:800], all_neg_x[:800])), np.vstack((all_pos_y[:800], all_neg_y[:800]))
    test_x, test_y = np.vstack((all_pos_x[800:], all_neg_x[800:])), np.vstack((all_pos_y[800:], all_neg_y[800:]))

    #My computer isn't powerful enough and I kept running out of memory :'(
    all_pos_x, all_pos_y, all_neg_x, all_neg_y = None, None, None, None #To free memory 

    #training
    print("[*] Training dataset...")
    weights = np.zeros(shape=(1, len(bow)))
    new_weights = train(train_x, train_y, weights)

    #test
    print("[*] Testing dataset...")
    percent_acc = test(test_x, test_y, new_weights[-1])
    print("Training accuracy: %.2f" %(percent_acc))
    new_weights = None #Once again, memory freeing

    #Perform training using randomized training data with seed
    print("[*] Training randomized dataset with seed 180128251")
    data_index = [x for x in range(train_y.shape[0])] #Create list to contain all indices of training data
    random.seed(180128251)

    #Shuffle index list and assign new random daa based on list
    random.shuffle(data_index)
    random_x = train_x[data_index]
    random_y = train_y[data_index]

    #training
    print("[*] Training dataset...")
    new_weights = train(random_x, random_y, weights)

    #test
    print("[*] Testing dataset...")
    percent_acc = test(test_x, test_y, new_weights[-1])
    print("Training accuracy: %.2f" %(percent_acc)) 
    new_weights = None #Once again, memory freeing

    #Multiple passes over training data
    print("[*] Training over multiple passes")
    num_iters = 10
    training_progress = []

    #Create list of all weights
    weights = [np.zeros(shape=(1, len(bow)))]
    for i in range(num_iters):
        #training
        random.shuffle(data_index)
        random_x = train_x[data_index]
        random_y = train_y[data_index]

        #Add to list of all weights 
        weights.extend(train(random_x, random_y, weights[-1]))
        percent_acc = test(test_x, test_y, weights[-1])

        print("[*] Training iteration: %d Accuracy so far: %.2f" %(i, percent_acc))
        training_progress.append(percent_acc)   

    #Compute average of weights
    avg_weight = sum(weights)/(num_iters*train_x.shape[0])

    print("Unigram x: ", [i for i in range(10)], " accuracy: ", training_progress)
    plt.plot([i for i in range(10)], training_progress)
    plt.xlabel("Iteration number")
    plt.ylabel("Training accuracy")

    plt.show()

    #Training accuracy using average of all weight vectors
    percent_acc_avg = test(test_x, test_y, avg_weight)
    print("[*] Training accuracy using average of all weights: %.2f" %(percent_acc_avg))

    bow_index = [c for c in range(len(bow))]  #Getting indices for all bag of words

    #Sort indices by average weights to obtain words from 
    bow_index.sort(key=lambda i: avg_weight.tolist()[0][i]) 

    #Get most sentimentally negativie words
    print("\nTop 10 weighted features for neg class")
    for x in range(10):
        print("Word: %s, Weight: %.2f" %(bow[bow_index[x]], avg_weight[0][bow_index[x]]))

    #Get most sentimentally positive words
    bow_index.reverse()
    print("\nTop 10 weighted features for pos class")
    for x in range(10):
        print("Word: %s, Weight: %.2f" %(bow[bow_index[x]], avg_weight[0][bow_index[x]]))


for i in range(1,4):
    print("\n\nRunning application with %d-gram" %i)
    run_program(i)
    time.sleep(5)