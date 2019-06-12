from collections import Counter
import sys, re
import numpy as np

if len(sys.argv) != 3:
	print("[*] Usage: python %s <corpus> <questions>" %sys.argv[0])
	sys.exit(0)

context = [] #Hold unigram model
word_context = [] #Hold bigram model
with open(sys.argv[1], "r") as fh:
	for line in fh:
		words = ["<s>"]+[w for w in line.lower().split(" ") if re.search('\w', w)]+["</s>"]
		context += words #Create unigram model
		word_context += [" ".join([words[i+j] for j in range(2)]) for i in range(len(words)-1)] #Create bigram model

context = Counter(context) #Get count for unigram model
word_context = Counter(word_context) #Get count for bigram model
total_words = sum(list(context.values())) #Get total number of words in corpus

#Compute probability value for word using unigram model
def unigram(value):
	return context.get(value, 0)/total_words

#Compute probability value for context+word using bigram model with optional smoothing
def bigram(value, smoothing=0):
	try:
		return (word_context.get(value, 0)+smoothing)/(context.get(value.split(" ")[0], 0)+smoothing*(len(context)))
	except ZeroDivisionError:
		return 0

#Obtain prediction from probabilities and answers
def predict(p1, p2, a1, a2):
	if p1 > p2:
		return a1, p1, False
	elif p1 < p2:
		return a2, p2, False
	else:
		return a1, p2, True

#Get right answers for measuring accuracy
ra = ["whether", "through", "piece", "court", "allowed", "check", "hear", "cereal", "chews", "sell"]
ai, ca_u, ca_b, ca_ba = 0, 0, 0, 0 #Accuracy value
with open(sys.argv[2], "r") as fh:
    for line in fh:
        u1, u2, b1, b2, ba1, ba2 = 1, 1, 1, 1, 1, 1 #Probability values for different models
        answer1, answer2 = line.split(":")[1].split("/")[0].strip(), line.split(":")[1].split("/")[1].strip() #Get answers from question set
        line = line.split(":")[0] 
        words = ["<s>"]+[w for w in line.lower().split(" ") if re.search('\w', w)]+["</s>"] #Get unigram test words
        bi_words = [" ".join([words[i+j] for j in range(2)]) for i in range(len(words)-1)] #Get bigram test words

        #Compute probabilities for unigram
        for word in words:
            u1 *= unigram(word.replace("____", answer1))
            u2 *= unigram(word.replace("____", answer2))

        #Compute probabilies for bigram (with add-1 smoothing)
        for bi_word in bi_words:
            b1 *= bigram(bi_word.replace("____", answer1))
            b2 *= bigram(bi_word.replace("____", answer2))

            ba1 *= bigram(bi_word.replace("____", answer1), 1)
            ba2 *= bigram(bi_word.replace("____", answer2), 1)

        #Get predicted answers and print values.
        print("Unigram model: ", end="")
        ans, prob, tie = predict(u1, u2, answer1, answer2)
        if tie == True:
            if prob == 0: print("Incorrect Answer")
            else: 
                ca_u += 0.5
                print(ans)
        else:
            if ans == ra[ai]: ca_u += 1
            print(ans)

        print("Bigram model: ", end="")
        ans, prob, tie = predict(b1, b2, answer1, answer2)
        if tie == True:
            if prob == 0: print("Incorrect Answer")
            else: 
                ca_b += 0.5
                print(ans)
        else:
            if ans == ra[ai]: ca_b += 1
            print(ans)


        print("Bigram with add-1 smoothing model : ", end = "")
        ans, prob, tie = predict(ba1, ba2, answer1, answer2)
        if tie == True:
            if prob == 0: print("Incorrect Answer")
            else: 
                ca_ba += 0.5
                print(ans)
        else:
            if ans == ra[ai]: ca_ba += 1
            print(ans)

        ai+=1

        print("\n ")

#Copmute and pring accuracy for different models
print("Unigram percentage accuracy: ", (ca_u/len(ra)) * 100)
print("Bigram percentage accuracy: ", (ca_b/len(ra)) * 100)
print("Bigram with add-1 smoothing percentage accuracy: ", (ca_ba/len(ra)) * 100)