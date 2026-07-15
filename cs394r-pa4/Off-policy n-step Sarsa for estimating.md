# Off-policy n-step Sarsa for estimating $\bar{Q} \approx q_*$, or $q_\pi$

**Inputs:**
- Behavior policy $b$ such that $b(a|s) > 0$ for all $s \in \mathcal{S}, a \in \mathcal{A}$
- Step size $\alpha \in (0, 1]$
- Positive integer $n$

**Initialization:**
$$Q(s, a) \text{ arbitrarily, for all } s \in \mathcal{S}, a \in \mathcal{A}$$
$$\pi \text{ initialized to be greedy w.r.t. } Q, \text{ or as a fixed given policy}$$

**Episode loop:**

1. **Initialize and store** $S_0 \neq \text{terminal}$

2. **Select and store** an action $A_0 \sim b(\cdot|S_0)$

3. **Loop** for $t = 0, 1, 2, \ldots$:

   If $t < T$:
   - Take action $A_t$
   - Observe and store next reward $R_{t+1}$ and next state $S_{t+1}$
   - If $S_{t+1}$ is terminal, then $T \leftarrow t + 1$
   - Else, select and store $A_{t+1} \sim b(\cdot|S_{t+1})$

4. Let $\tau = t - n + 1$ (the time step whose estimate is being updated)

5. **If** $\tau \geq 0$:

   **Importance sampling ratio:**
   $$\rho_{\tau+1:\min(\tau+n,\,T-1)} = \prod_{i=\tau+1}^{\min(\tau+n,\,T-1)} \frac{\pi(A_i|S_i)}{b(A_i|S_i)}$$

   **n-step return:**
   $$G = \sum_{i=\tau+1}^{\min(\tau+n,\,T)} \gamma^{\,i-\tau-1} R_i$$

   **Bootstrap** (if episode hasn't ended):
   $$\text{If } \tau + n < T: \quad G \leftarrow G + \gamma^n Q(S_{\tau+n}, A_{\tau+n})$$

   **Update:**
   $$Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha \, \rho \,\bigl[\,G - Q(S_\tau, A_\tau)\,\bigr]$$

   **Policy improvement** (if learning $\pi$):
   Ensure $\pi(\cdot|S_\tau)$ is greedy w.r.t. $Q$

6. **Until** $\tau = T - 1$