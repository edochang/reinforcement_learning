# n-step TD for Estimating $v_\pi$

**Input:** a policy $\pi$

**Algorithm parameters:** step size $\alpha \in (0, 1]$, a positive integer $n$

1. Initialize $V(s)$ arbitrarily, for all $s \in \mathcal{S}$
2. All store and access operations (for $S_t$ and $R_{t+1}$) can take their index modulo $n + 1$

**Loop for each episode:**

1. Initialize and store $S_0 \neq \text{terminal}$
   
2. $T \leftarrow \infty$
   
3. **Loop for** $t = 0, 1, 2, \dots$:
    
    **If** $t < T$:
    - Take an action according to $\pi(\cdot \mid S_t)$
    - Observe and store the next reward as $R_{t+1}$ and the next state as $S_{t+1}$
    - **If** $S_{t+1}$ is terminal, then $T \leftarrow t + 1$
      
    $\tau \leftarrow t - n + 1$ &nbsp;&nbsp;($\tau$ is the time whose state's estimate is being updated)
      
    **If** $\tau \geq 0$:
    
    - $G \leftarrow \displaystyle\sum_{i=\tau+1}^{\min(\tau+n, T)} R_{i-1}$
    - **If** $\tau + n < T$: then $G \leftarrow G + V(S_{\tau+n})$ &nbsp;&nbsp;($G^{t:\tau+n}$)
    - $V(S_\tau) \leftarrow V(S_\tau) + \alpha \big[G - V(S_\tau)\big]$
   
4. **Until** $\tau = T - 1$