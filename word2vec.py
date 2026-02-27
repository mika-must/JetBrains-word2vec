import numpy as np
import os
import re
from collections import Counter

#First we need a tokenizer

def tokenize(text: str) ->list[str]:
    """
    We need to lowercase our text and extract tokens consisting of letters, digits and apostrophes (simple).
    """
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)

#Now we need to build a vocabulary from our tokens

def build_vocab(tokens: list[str], min_count: int = 2):
    """
    We do that by:
    1. Counting token frequencies
    2. Keeping tokens with count >= min_count
    3. Assigning ids [0...V-1]
    Returns: word_to_id, id_to_word, counts
    """

    counter = Counter(tokens)

    #tokens that are frequent enough
    vocabulary = [w for w, c in counter.items() if c >= min_count]
    
    #sort: descending by frequency, alphabetic at tie
    vocabulary.sort(key = lambda w: (-counter[w], w))


    word_to_id = {w: i for i, w in enumerate (vocabulary)}
    id_to_word = vocabulary
    counts = np.array([counter[w] for w in vocabulary], dtype = np.int64)
    return word_to_id, id_to_word, counts


#The next step is to convert tokens into ids

def tokens_to_ids(tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
    """
    Convert tokens to a np array of word ids, while dropping tokens not found in word_to_id
    """
    ids = [word_to_id[w] for w in tokens if w in word_to_id]
    return np.array(ids, dtype = np.int32)

#Now that we have the corpus as integers we can implement skipgram pairs

def skipgram_pairs(ids: np.ndarray, window: int, rng:np.random.Generator):
    """
    Uses a dynamic window (sample w in [1, window]) and yields pairs of center_id and context_id
    """
    N = len(ids)
    for i in range(N):
        center_id = int(ids[i])

        w = int(rng.integers(1, window + 1))

        left = max(0, i-w)
        right = min(N, i+w+1)

        for j in range(left, right):
            if j == i:
                continue
            context_id = int(ids[j])
            yield center_id, context_id




#I decided to use negative sampling, so here is the implementation

def negative_sampling_dist(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """
    p_ neg over the vocabulary:
    p_neg[i] = counts[i]^power / sum_j counts[j]^power
    """
    p = counts.astype(np.float64) ** power
    p/= p.sum()
    return p


#Next, we sample K negative word ids
def sample_negatives(p_neg: np.ndarray, K:int, forbidden_id:int, rng: np.random.Generator) -> np.ndarray:
    V = len(p_neg)
    negs = []

    while len(negs) < K:
        need = K - len(negs)

        cand = rng.choice(V, size=need * 2, replace=True, p=p_neg)

        cand = cand[cand != forbidden_id]

        negs.extend(cand[:need].tolist())

    return np.array(negs, dtype=np.int32)

#Now we can initialize embeddings

def init_embeddings(V: int, D: int, rng: np.random.Generator):
    """
    Returns W_in:  (V, D) and W_out: (V, D)
    """
    W_in  = (rng.normal(0.0, 1.0, size=(V, D)).astype(np.float32) / np.sqrt(D))
    W_out = (rng.normal(0.0, 1.0, size=(V, D)).astype(np.float32) / np.sqrt(D))
    return W_in, W_out

#Sigmoid function with numerical stability
def sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))

#SGD update 
def train_on_pair(center_id: int,
                  context_id: int,
                  W_in: np.ndarray,
                  W_out: np.ndarray,
                  p_neg: np.ndarray,
                  K: int,
                  lr: float,
                  rng: np.random.Generator):
    #Sample negatives
    V = len(p_neg)
    neg_ids = []
    neg_ids = sample_negatives(p_neg, K=K, forbidden_id=context_id, rng=rng)  

    #Gather vectors
    v = W_in[center_id]                 
    ids = np.concatenate(([context_id], neg_ids)) 
    U = W_out[ids] 

    #Forward pass
    scores = U @ v

    
    labels = np.zeros(K + 1, dtype=np.float32)
    labels[0] = 1.0

    #Logistic Loss per item
    #softplus(score) - y*score which equals -log(sigmoid(score)) for y=1 and -log(sigmoid(-score)) for y=0)
    loss_vec = np.logaddexp(0.0, scores) - labels * scores
    loss = float(loss_vec.sum())

    #Backward pass - Gradient with respect to scores dL/dscore = sigmoid(score) - y
    probs = sigmoid(scores)              
    grad_scores = probs - labels

    #Gradients with respect to embeddings
    #score_i = U_i dot v
    #dL/dv = sum_i grad_scores[i] * U_i
    grad_v = grad_scores @ U

    #dL/dU_i = grad_scores[i] * v
    grad_U = np.outer(grad_scores, v)

    #SGD update
    W_in[center_id] -= lr * grad_v.astype(np.float32)

    #Handling of potentially repeated ids
    np.add.at(W_out, ids, (-lr * grad_U).astype(np.float32))

    return loss

#FInally wrap everything into a training loop

def train_sgns(ids: np.ndarray,
               W_in: np.ndarray,
               W_out: np.ndarray,
               p_neg: np.ndarray,
               window: int,
               K: int,
               lr: float,
               epochs: int,
               rng: np.random.Generator,
               log_every: int = 100_000):
    
    epoch_avg_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_updates = 0

        for center_id, context_id in skipgram_pairs(ids, window=window, rng=rng):
            loss = train_on_pair(center_id=center_id,
                                 context_id=context_id,
                                 W_in=W_in,
                                 W_out=W_out,
                                 p_neg=p_neg,
                                 K=K,
                                 lr=lr,
                                 rng=rng)

            total_loss += loss
            n_updates += 1

            if log_every and (n_updates % log_every == 0):
                print(f"[epoch {epoch}] updates={n_updates:,} avg_loss={total_loss/n_updates:.4f}")

        avg_loss = total_loss / max(n_updates, 1)
        epoch_avg_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} finished | updates={n_updates:,} | avg_loss={avg_loss:.4f}")

    return W_in, W_out, epoch_avg_losses

#Wrap everything into a single function

def fit_word2vec_sgns(text: str,
                      D: int = 100,
                      window: int = 5,
                      K: int = 5,
                      lr: float = 0.03,
                      epochs: int = 3,
                      min_count: int = 2,
                      seed: int = 0):
    rng = np.random.default_rng(seed)

    tokens = tokenize(text)
    word_to_id, id_to_word, counts = build_vocab(tokens, min_count=min_count)
    ids = tokens_to_ids(tokens, word_to_id)

    V = len(id_to_word)
    p_neg = negative_sampling_dist(counts)
    W_in, W_out = init_embeddings(V, D, rng)

    # uses your previously-defined skipgram_pairs, train_on_pair, train_sgns
    W_in, W_out, losses = train_sgns(ids=ids,
                                     W_in=W_in,
                                     W_out=W_out,
                                     p_neg=p_neg,
                                     window=window,
                                     K=K,
                                     lr=lr,
                                     epochs=epochs,
                                     rng=rng)

    return word_to_id, id_to_word, W_in, W_out, losses

#An additional function to test the code out 
def most_similar(word: str, word_to_id, id_to_word, W_in, W_out, topn=5):
    if word not in word_to_id:
        raise KeyError(f"{word} not in vocab")

    emb = (W_in + W_out) / 2.0
    v = emb[word_to_id[word]].astype(np.float64)

    v_norm = np.linalg.norm(v) + 1e-9
    E = emb.astype(np.float64)
    E_norm = np.linalg.norm(E, axis=1) + 1e-9

    sims = (E @ v) / (E_norm * v_norm)
    sims[word_to_id[word]] = -np.inf

    best = np.argsort(-sims)[:topn]
    return [(id_to_word[i], float(sims[i])) for i in best]

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    CORPUS_PATH = os.path.join(HERE, "corpus.txt")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
        word_to_id, id_to_word, W_in, W_out, losses = fit_word2vec_sgns(
        text, D=100, window=5, K=5, lr=0.03, epochs=2, min_count=1, seed=0 #Decided to set min_count to 1 in the end because of a smaller corpus
    )
    print("similar to 'gom':", most_similar("gom", word_to_id, id_to_word, W_in, W_out))
    print("losses:", losses)
    print("Vocab size:", len(id_to_word))