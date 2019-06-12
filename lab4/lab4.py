from collections import Counter, defaultdict
import numpy as np
from itertools import product
from random import shuffle, seed
from sys import argv, exit
from sklearn.metrics import f1_score
from argparse import ArgumentParser

parser = ArgumentParser(description="NLP Lab 4: Viterbi and Beam Search")

parser.add_argument("train_data", help="Training data", action="store")
parser.add_argument("test_data", help="Testing data", action="store")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--viterbi", help="Use viterbi", action="store_true", default=False)
group.add_argument("-b", "--beam", help="Use beam search", action="store", type=int, default=None) 

args = parser.parse_args()

#Transform data into more presentable format
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()] 
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


#Get current word-current label frequency count for corpus
def cw_cl_count(corpus, threshold=3):
    cw_cl_count = []
    for s in corpus:
        for cw_cl in s:
            cw_cl_count.append(cw_cl[0]+"_"+cw_cl[1])

    return {cw_cl: count for cw_cl, count in Counter(cw_cl_count).items() if count >= threshold}

#Phi-1 feature set to obtain cw-cl-count for sentence and corresponding label
def phi_1(x, y, cw_cl_counts):
    cw_cl_counts_s = {}
    for x_i, y_i in zip(x, y):
        if x_i+"_"+y_i in cw_cl_counts:
            cw_cl_counts_s[x_i+"_"+y_i] = cw_cl_counts_s.get(x_i+"_"+y_i, 0) + 1

    return cw_cl_counts_s

#Same as cw_cl_count but for previous label-current label 
def pl_cl_count(corpus, threshold=3):
    pl_cl_count = []
    for s in corpus:
        s = [("None", "None")] + s[:]
        for i in range(len(s)-1):
            pl_cl_count.append(s[i][1]+"_"+s[i+1][1])

    return {pl_cl: count for pl_cl, count in Counter(pl_cl_count).items() if count >= threshold}

#Same as phi_1 but for pl_cl_count
def phi_2(x, y, pl_cl_counts):
    pl_cl_counts_s = {}
    y = ["None"] + list(y[:])
    for i in range(len(y)-1):
        if y[i]+"_"+y[i+1] in pl_cl_counts:
            pl_cl_counts_s[y[i]+"_"+y[i+1]] = pl_cl_counts_s.get(y[i]+"_"+y[i+1], 0) + 1

    return pl_cl_counts_s

#Combine phi1+phi2
phi_combo = lambda phi1, phi2: {**phi1, **phi2}

#Argmax function to find best label from list of all possible labels
def argmax(w, x, feat_type, y_N):
    pred_y_N = {}
    y_N = list(y_N)
    for i, y in enumerate(y_N):
        phi = phi_1(x, y, feat_type) if not isinstance(feat_type, list) else phi_combo(phi_1(x, y, feat_type[0]), phi_2(x, y, feat_type[1]))
        pred_y_N[i] = sum([w.get(ft, 0)*c for ft, c in phi.items()])
    
    return list(y_N)[max(pred_y_N, key=lambda x: pred_y_N[x])]

#Viterbi algorithm function
def viterbi(w, x, feat_type):
    pred_y_N = [(None, 0)]
    pred_y = []
    ner_tags = ["PER", "O", "LOC", "ORG", "MISC"]
    for x_i in x:
        #Loop for every sentence_tag combination and set unavailables to 0
        phi = phi_1([x_i for i in range(5)], ner_tags, feat_type)
        phi = {x_i+"_"+tag:phi[x_i+"_"+tag] if x_i+"_"+tag in phi else 0 for tag in ner_tags}

        #Loop through the previous tag values and assign, the current word_label to the maximum viterbi score. (Perform this for every word_label) type and sort
        pred_y_N = sorted([(k.split("_")[-1], max([py[1]+v*w.get(k, 0) for py in pred_y_N])) for k, v in phi.items()], key=lambda p: p[1], reverse=True)
        pred_y.append(pred_y_N[0][0])

    return pred_y

#Beam search algorithm function
def beam_search(w, x, feat_type, top_k=3):
    pred_y_N = [(None, 0)]
    ner_tags = ["PER", "O", "LOC", "ORG", "MISC"]
    pred_y = []
    for x_i in x:
        phi = phi_1([x_i for i in range(5)], ner_tags, feat_type)
        phi = {x_i+"_"+tag:phi[x_i+"_"+tag] if x_i+"_"+tag in phi else 0 for tag in ner_tags}

        #Same as viterbi but return top_k
        pred_y_N = sorted([(k.split("_")[-1], max([py[1]+v*w.get(k, 0) for py in pred_y_N])) for k, v in phi.items()], key=lambda p: p[1], reverse=True)[:top_k]
        pred_y.append(pred_y_N[0][0])

    return pred_y


#Training function -> D - [([x_1*],[y_1*]), ([x_2*],[y_2*]), ..., ([x_n], [y_n])]
def train(D, w, pfi=1):
    feat_type = [cw_cl_count(D, 1), pl_cl_count(D, 1) if pfi != 1 else None]

    w = defaultdict(lambda: 0) if w == None else w
    #Looping through training -> x - [w_1, w_2, ..., w_n] ; y - [l_1, l_2, ..., l_n]
    for x_y in D:
        #Predict y
        x = [z[0] for z in x_y]
        y = [z[1] for z in x_y]

        
        if args.viterbi:
            y_pred = viterbi(w, x, feat_type[0])
        elif args.beam is not None:
            y_pred = beam_search(w, x, feat_type[0], args.beam)
        else: 
            y_pred = argmax(w, x, feat_type[0] if pfi == 1 else feat_type, product(["PER", "O", "LOC", "ORG", "MISC"], repeat=len(x)))
        
        if y_pred != y:
            phi_c = phi_1(x, y, feat_type[0]) 
            if pfi == 2: phi_c = phi_combo(phi_c, phi_2(x, y, feat_type[1]))
            
            phi_pred = phi_1(x, y_pred, feat_type[0]) 
            if pfi == 2: phi_pred = phi_combo(phi_pred, phi_2(x, y_pred, feat_type[1]))
    
            for ft in set(list(phi_c.keys()) + list(phi_pred.keys())):
                w[ft] = w.get(ft, 0) + phi_c.get(ft, 0) - phi_pred.get(ft, 0)
            
    return w

#Prediction function (Just a wrapper for argmax)
if args.viterbi:
    predict = lambda x, feat_type, w: viterbi(w, x, feat_type)
elif args.beam is not None:
    predict = lambda x, feat_type, w: beam_search(w, x, feat_type, args.beam)
else:
    predict = lambda x, feat_type, w: argmax(w, x, feat_type, product(["PER", "O", "LOC", "ORG", "MISC"], repeat=len(x)))
#Test function - returns f1-micro score of prediction
def test(test_data, feat_type, w):
    y_correct = []
    y_predict = []
    for x_y in test_data:
        x = [z[0] for z in x_y]
        y_correct += [z[1] for z in x_y]
        y_predict += predict(x, feat_type, w)
    return f1_score(y_correct, y_predict, average="micro", labels=["ORG", "MISC", "PER", "LOC"])

#Average function for finding average of weight values
avg_weights = lambda weights, n_div: {k: sum([z[k] for z in weights])/(n_div) for k in weights[0].keys()}

#Get training data
train_data = load_dataset_sents(args.train_data)
test_data = load_dataset_sents(args.test_data)

seed(180128251) 

##### PHI 1 #####
num_iters = 5
weights = [None]
print("[*] Training with %d iterations using phi_1" %num_iters)
for i in range(num_iters):
    shuffle(train_data)
    weights.append(dict(sorted(train(train_data, weights[-1], 1).items())))
average_weights = avg_weights(weights[1:], len(train_data)*num_iters)
f1_score_v = test(test_data, cw_cl_count(train_data, 1), average_weights)
print("[*] F1-Score for phi_1 feature set: %.5f" %f1_score_v)
average_weights = sorted(average_weights.items(), key=lambda x: x[1], reverse=True)

#Top 10 weight features for different labels
print("Top 10 weighted features for ORG: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_ORG")][:10]))
print("Top 10 weighted features for PER: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_PER")][:10]))
print("Top 10 weighted features for LOC: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_LOC")][:10]))
print("Top 10 weighted features for MISC: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_MISC")][:10]))
print("Top 10 weighted features for O: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_O")][:10]))

print()
##### PHI 1 + PHI 2 #####
if not args.viterbi and args.beam is None:
    weights = [None]
    print("[*] Training with %d iterations using phi_1+phi_2" %num_iters)
    for i in range(num_iters):
        shuffle(train_data)
        weights.append(dict(sorted(train(train_data, weights[-1], 2).items())))
    average_weights = avg_weights(weights[1:], len(train_data)*num_iters)
    f1_score_v = test(test_data, [cw_cl_count(train_data, 1), pl_cl_count(train_data, 1)], average_weights)
    print("[*] F1-Score for phi_1+phi_2 feature set: %.5f" %f1_score_v)
    average_weights = sorted(average_weights.items(), key=lambda x: x[1], reverse=True)

    #Top 10 weight values for different labels
    print("Top 10 weighted features for ORG: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_ORG")][:10]))
    print("Top 10 weighted features for PER: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_PER")][:10]))
    print("Top 10 weighted features for LOC: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_LOC")][:10]))
    print("Top 10 weighted features for MISC: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_MISC")][:10]))
    print("Top 10 weighted features for O: ", ", ".join([k[0] for k in average_weights if k[0].endswith("_O")][:10]))
