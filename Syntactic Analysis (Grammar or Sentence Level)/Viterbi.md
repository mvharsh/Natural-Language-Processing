# VITERBI ALGORITHM

## AIM

To implement the Viterbi algorithm for Hidden Markov Models (HMMs) in order to find the most probable sequence of POS tags (hidden states) for a given sequence of words (observations).

---

## ALGORITHM

1. **Initialization (t = 0):**

   * For each state `s`:
     `V[0][s] = start_p[s] * emit_p[s][obs[0]]`
   * `Path[s] = [s]`

2. **Recursion (for t = 1 … T-1):**

   * For each current state `curr`:

     * Compute
       `max(V[t-1][prev] * trans_p[prev][curr] * emit_p[curr][obs[t]])`
     * Store best probability in `V[t][curr]`
     * Extend best path with `curr`

3. **Termination:**

   * Select final state with maximum `V[T-1][s]`
   * Return its probability and the corresponding path

---

## PROGRAM

```python
states = ['Noun', 'Verb']

observations = ['time', 'flies', 'like', 'an', 'arrow']

start_probability = {'Noun': 0.6, 'Verb': 0.4}

transition_probability = {
    'Noun': {'Noun': 0.1, 'Verb': 0.9},
    'Verb': {'Noun': 0.8, 'Verb': 0.2}
}

emission_probability = {
    'Noun': {'time': 0.5, 'flies': 0.1, 'like': 0.05, 'an': 0.05, 'arrow': 0.3},
    'Verb': {'time': 0.1, 'flies': 0.6, 'like': 0.3, 'an': 0.0, 'arrow': 0.0}
}

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # Stores the probabilities
    path = {}  # Stores the best path

    # Initialize base cases (t == 0)
    for s in states:
        V[0][s] = start_p[s] * emit_p[s].get(obs[0], 0)
        path[s] = [s]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for curr_state in states:
            (prob, prev_state) = max(
                (V[t - 1][prev_state] * trans_p[prev_state][curr_state] * emit_p[curr_state].get(obs[t], 0), prev_state)
                for prev_state in states
            )
            V[t][curr_state] = prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Find the final most probable state
    n = len(obs) - 1
    (prob, state) = max((V[n][s], s) for s in states)
    return (prob, path[state])

prob, pos_tags = viterbi(observations, states, start_probability, transition_probability, emission_probability)

print("Best POS Tag Sequence:", pos_tags)
print("Probability of the best path:", prob)
```

---

## OUTPUT

```
Best POS Tag Sequence: ['Noun', 'Verb', 'Verb', 'Noun', 'Noun']
Probability of the best path: 1.1664000000000004e-05
```

---
