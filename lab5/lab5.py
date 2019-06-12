# Author: Robert Guthrie
# Edited: 180128251

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 3

sentences = ["The mathematician ran .", "The mathematician ran to the store .", "The physicist ran to the store .", "The philosopher thought about it .", "The mathematician solved the open problem ."]

trigrams_sentences = []
for sent in sentences:
    sent = ["<s>"] + sent.split() + ["</s>"]
    sent_tri = []
    for i in range(len(sent)-2):
        sent_tri.append(([sent[i], sent[i+1]], sent[i+2]))
    trigrams_sentences.append(sent_tri)

vocab = set((' '.join(["<s>"] + sentences + ["</s>"])).split())
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs



losses = []
loss_function = nn.NLLLoss() #Define negative log likelihood loss function
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.03)

for epoch in range(400):
    total_loss = torch.Tensor([0])
    for sent_tri in trigrams_sentences:
        for context, target in sent_tri:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_var)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
    losses.append(total_loss)
print("Most recently loss: %f" %losses[-1])  # The loss decreased every iteration over the training data!

#running sanity check
ss = "<s> The mathematician ran to the store . </s>".split()
for context, target in [([ss[i], ss[i+1]], ss[i+2]) for i in range(len(ss) - 2)]:

    context_idxs = [word_to_ix[w] for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_idxs))
    log_probs = model(context_var)

    v_id = torch.argmax(log_probs)

    print("Context: %s |=> prediction: %s" %(' '.join(context), list(vocab)[v_id]))
 
print("\n\n")
#test our model
ts = "<s> The _____ solved the open problem . </s>".split()
gap_index = ts.index("_____")

#get only parts of sentence with gap in it
# ts = ts[gap_index-CONTEXT_SIZE: gap_index+CONTEXT_SIZE+1]

ma_i = list(vocab).index("mathematician")
ps_i = list(vocab).index("physicist")
ph_i = list(vocab).index("philosopher")
prob_v = [1, 1]
for j, pred_w in enumerate(["physicist", "philosopher"]):
    for context, target in [([ts[i], ts[i+1]], ts[i+2]) for i in range(len(ts) - 2)]:

        context_idx = [word_to_ix[w] if w != "_____" else word_to_ix[pred_w] for w in context]
    
        context_var = autograd.Variable(torch.LongTensor(context_idx))

        #calculate sentence probability
        log_probs = model(context_var).detach()
        # prob_v[j] += np.where(np.argsort(log_probs.numpy())[0] == list(vocab).index(target.replace("_____", pred_w)))[0][0]
        prob_v[j] *= log_probs.numpy()[0][list(vocab).index(target.replace("_____", pred_w))]
    
#word with less positional probability is less more likely target word
print("[*] Prediction from model is %s" %("physicist" if prob_v[0] > prob_v[1] else "philosopher"))

#compute cosine similarity of words
math_T = model.embeddings(torch.LongTensor([ma_i]))
phy_T = model.embeddings(torch.LongTensor([ps_i]))
phi_T = model.embeddings(torch.LongTensor([ph_i]))
print("Cosine similarity between mathematician and philosopher: %f" %(F.cosine_similarity(phi_T, math_T, dim=-1)))
print("Cosine similarity between mathematician and physicist: %f" %(F.cosine_similarity(phy_T, math_T, dim=-1)))
