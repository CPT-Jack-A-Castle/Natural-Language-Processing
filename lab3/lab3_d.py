from collections import Counter
import numpy as np
import random
from random import shuffle
from sklearn.metrics import f1_score
import sys
from itertools  import product


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None): 
    '''
    Function to load data from train/test files
    '''
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split("\t")
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()] 
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

def cw_cl_counter(corpus, min_counts = 3):
    '''
    Current word-current label dictionary
    It doesn't include features with a less tham min_counts frequency, default: 3
    '''
    cw_cl = []
    for s in corpus:
        cw_cl += [word+"_"+label for word,label in s]
    counter = Counter(cw_cl)
    return {k:v for k,v in counter.items() if v >= min_counts}

def phi_1(x, y, cw_cl_counts):
    '''
    Given a sentence x and a sequence y, returns the counts for each
    current word-current label feature in the sentence that is also in the corpus (cw_cl_counts)
    '''
    features = []
    for i in range(len(x)):
        f = x[i]+"_"+y[i]
        if f in cw_cl_counts:
            features.append(x[i]+"_"+y[i])
    return Counter(features)

def pl_cl_counter(corpus, min_counts = 3):
    '''
    Previous label-current label dictionary
    It doesn't include features with a less tham min_counts frequency, default: 3
    '''
    pl_cl = []
    for s in corpus:  
        pl_cl += [s[i-1][1]+"_"+s[i][1] for i in range(1,len(s))]
    counter = Counter(pl_cl) 
    return {k:v for k,v in counter.items() if v > min_counts}
    

def phi_2(x,y, pl_cl_counts):
    '''
    Given a sentence x and a sequence y, returns the counts for each
    previous label-current label feature in the sentence that is also in the corpus (pl_cl_counts)
    '''
    features = []
    for i in range(1,len(y)):
        k = y[i-1]+"_"+y[i]
        if k in pl_cl_counts:
            features.append(k)
    return Counter(features)

def pfw_cl_counter(corpus, min_counts = 3):
    '''
    Previous first letter of current word-current label dictionary
    It doesn't include features with a less tham min_counts frequency, default: 3
    '''
    pfw_cl = []
    for s in corpus:
        pfw_cl += [word[0]+"_"+label for word,label in s]
    counter = Counter(pfw_cl)
    return {k:v for k,v in counter.items() if v > min_counts}

def phi_3(x,y, pfw_cl_counts):
    '''
    Given a sentence x and a sequence y, returns the counts for each
    previous first letter of current word-current label feature in the sentence that is also in the corpus (pfw_cl_counts)
    '''
    features = []
    for i in range(1,len(x)):
        k = x[i][0]+"_"+y[i]
        if k in pfw_cl_counts:
            features.append(k)
    return Counter(features)

def cs_cl_counter(corpus, min_counts = 3):
    '''
    Current 3 letter suffix-current label dictionary
    It doesn't include features with a less tham min_counts frequency, default: 3
    '''
    cs_cl = []
    for s in corpus:
        cs_cl += [word[-3:]+"_"+label for word,label in s]
    counter = Counter(cs_cl) 
    return {k:v for k,v in counter.items() if v > min_counts} 

def phi_4(x,y, cs_cl_counts):
    '''
    Given a sentence x and a sequence y, returns the counts for each
    current 3 letter suffix-current feature in the sentence that is also in the corpus (cs_cl_counts)
    '''
    features = []
    for i in range(len(x)):
        k = x[i][-3:]+"_"+y[i]
        if k in cs_cl_counts:
            features.append(k)
    return Counter(features)


def phi(x,y,phi_counts):
    '''
    Given a sentence x, a sequence y, and a dictionary phi_counts ({phi_functions:counter_functions})
    returns a combined dictionary of features
    '''
    features = {}
    for phi_func,counts in phi_counts.items():
        #I'm gonna replace this with what's below, I think it'll make it faster
        # for feature, count in phi_func(x,y,counts).items():
        #     features[feature] = count
        features.update(phi_func(x,y,counts)) # <- Wuzi code
    return features

#Looks good
def perceptron(corpus,w,c,phi_counts):
    '''
    Given a copus, a w (dict), a number of updates c (int) and a phi_counts ({phi_functions:counter_functions})
    performs an iteration of the perceptron
    '''
    for k,s in enumerate(corpus):
        # initialize variables
        x = [i[0] for i in s]
        y_target = [i[1] for i in s]
        y_pred = predict(x,w,phi_counts)
        if y_pred != y_target:
            # w = w + phi(x,y ) - phi(x ,y_pred)
            phi_diff = Counter(phi(x,y_target,phi_counts))
            phi_diff.subtract(phi(x,y_pred,phi_counts))
            for feature,count in phi_diff.items():
                if feature in w:
                    w[feature] += count
                else:
                    w[feature] = count
        c += 1
    return w,c

def train(corpus, phi_counts, num_iter = 10, averaging = True):
    '''
    Perform the perceptron training algorithm given a corpus, a phi_counts dictionary ({phi_functions:counter_functions}),
    a number of iterations num_iter (int) with/without averaging (boolean)
    '''
    c = 0
    w_c = {}
    w_avg = {}
    for iteration in range(num_iter):
        print("Iteration {}/{}".format(iteration+1,num_iter))
        # randomize
        shuffle(corpus)
        w_c,c = perceptron(corpus,w_c,c,phi_counts)
        if averaging:
            for feature, v in w_c.items():
                if feature in w_avg:
                    w_avg[feature] += v
                else:
                    w_avg[feature] = v
    if averaging:
        for f,v in w_avg.items():
            w_avg[f] = v/c
        return w_avg
    else:
        return w_c

#All good
def predict(x,w,phi_counts):
    '''
    Predicts a sequence y_pred given a sentence x, weights w (dict),
    and phi_counts dictionary ({phi_functions:counter_functions})
    '''
    max_score = -1e10
    y_pred = []
    # all possible sequences
    all_possible_y = product(labels, repeat = len(x))
    for y in all_possible_y:
        score = 0
        # if len(y) == len(x): This should always be true
        phi_xy = phi(x,y,phi_counts)
        # for each feature that is in phi and w (features>0)
        for feature in set(phi_xy).intersection(set(w)):
            score += phi_xy[feature]*w[feature]
        if score > max_score:
            max_score = score
            y_pred = y
    return y_pred

def test(w,corpus,phi_counts):
    '''
    Predicts a sequence for each sentence in the corpus and computes the f1score with
    the correct and predicted flatten sequences
    '''
    predicted, correct = [], []
    for s in corpus:
        x = [i[0] for i in s]
        correct += [i[1] for i in s]
        for y in predict(x,w,phi_counts):
            predicted.append(y)
    predicted = np.asarray(predicted)
    correct = np.asarray(correct)
    f1_micro = f1_score(correct, predicted, average = "micro",
                        labels = ['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro

def print_top10(w):
    sorted_w = sorted(w.items(), key=lambda d: d[1],reverse = True)
    print("General: ",", ".join([x[0] for x in sorted_w[:10]]))
    # take top 10 by tag
    tags = {label:[] for label in labels}
    for f,wval in sorted_w:
        lbl = f.split("_")[-1]
        if lbl in tags:
            tags[lbl].append(f)
    for tag,top in tags.items():
        print(tag+": "+", ".join(top[:10]))

# Commandline parameters
args = sys.argv[1:]
if len(args) == 2:
    train_file = args[0] 
    test_file = args[1] 
else:
    print("Please provide the train and test .txt files as: python3 lab3.py train.txt test.txt")
    sys.exit()

# set random seed
random.seed(30)

# Global variables
corpus = load_dataset_sents(train_file)
test_corpus = load_dataset_sents(test_file)
# computing training corpus counters of features
cw_cl_counts = cw_cl_counter(corpus)
pl_cl_counts = pl_cl_counter(corpus)
pfw_cl_counts = pfw_cl_counter(corpus)
cs_cl_counts = cs_cl_counter(corpus)
# possible labels in training set
labels = ['ORG', 'MISC', 'PER', 'LOC', 'O']

def main():
    '''
    # ................. Perceptron with 1 iteration, no averaging ............................
    print("Perceptron with 1 iteration, no averaging")
    # phi_1
    print("Phi1")
    phi_counts = {phi_1:cw_cl_counts}
    w_phi1 = train(corpus,phi_counts, num_iter = 1, averaging = False)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi1,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi1,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi1)
    print()
    # phi_1 + phi_2
    print("Phi1 + Phi2")
    phi_counts = {phi_1:cw_cl_counts,phi_2:pl_cl_counts}
    w_phi12 = train(corpus,phi_counts, num_iter = 1, averaging = False)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi12,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi12,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi12)
    print()
    # phi_1 + phi_2 + bonus
    print("Phi1 + Phi2 + bonus")
    phi_counts = {phi_1:cw_cl_counts, phi_2:pl_cl_counts,
                  phi_3:pfw_cl_counts, phi_4:cs_cl_counts}
    w_phi_bonus = train(corpus,phi_counts, num_iter = 1, averaging = False)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi_bonus,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi_bonus,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi12)
    print()
    # ................. Perceptron with 3 iterations and averaging ............................
    print("Perceptron with 3 iterations and averaging")
    # phi_1
    print("Phi1")
    phi_counts = {phi_1:cw_cl_counts}
    w_phi1 = train(corpus,phi_counts, num_iter = 3)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi1,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi1,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi1)
    print()
    # phi_1 + phi_2
    print("Phi1 + Phi2")
    phi_counts = {phi_1:cw_cl_counts,phi_2:pl_cl_counts}
    w_phi12 = train(corpus,phi_counts, num_iter = 3)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi12,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi12,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi12)
    print()
    # phi_1 + phi_2 + bonus
    print("Phi1 + Phi2 + bonus")
    phi_counts = {phi_1:cw_cl_counts, phi_2:pl_cl_counts,
                  phi_3:pfw_cl_counts, phi_4:cs_cl_counts}
    w_phi_bonus = train(corpus,phi_counts, num_iter = 3)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi_bonus,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi_bonus,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi_bonus)
    print()
    '''
    # ................. Perceptron with 5 iterations and averaging ............................
    print("Perceptron with 5 iterations and averaging")
    # phi_1
    print("Phi1")
    phi_counts = {phi_1:cw_cl_counts}
    w_phi1 = train(corpus,phi_counts, num_iter = 5)
    print("Testing...")
    print("F1 Score: train",round(test(w_phi1,corpus,phi_counts),3))
    print("F1 Score: test",round(test(w_phi1,test_corpus,phi_counts),3))
    print("Top 10 most positively-weighted features:")
    print_top10(w_phi1)
    print()

    # # phi_1 + phi_2
    # print("Phi1 + Phi2")
    # phi_counts = {phi_1:cw_cl_counts,phi_2:pl_cl_counts}
    # w_phi12 = train(corpus,phi_counts, num_iter = 5)
    # print("Testing...")
    # print("F1 Score: train",round(test(w_phi12,corpus,phi_counts),3))
    # print("F1 Score: test",round(test(w_phi12,test_corpus,phi_counts),3))
    # print("Top 10 most positively-weighted features:")
    # print_top10(w_phi12)
    # print()

    # # phi_1 + phi_2 + bonus
    # print("Phi1 + Phi2 + bonus")
    # phi_counts = {phi_1:cw_cl_counts, phi_2:pl_cl_counts,
    #               phi_3:pfw_cl_counts, phi_4:cs_cl_counts}
    # w_phi_bonus = train(corpus,phi_counts, num_iter = 5)
    # print("Testing...")
    # print("F1 Score: train",round(test(w_phi_bonus,corpus,phi_counts),3))
    # print("F1 Score: test",round(test(w_phi_bonus,test_corpus,phi_counts),3))
    # print("Top 10 most positively-weighted features:")
    # print_top10(w_phi_bonus)
    

if __name__ == "__main__":
    main()
