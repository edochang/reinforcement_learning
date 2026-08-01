# True online Sarsa($\lambda$) for estimating $\mathbf{w}^\top \mathbf{x} \approx q_\pi$ or $q_*$

**Input:** a feature function $\mathbf{x} : \mathcal{S}^+ \times \mathcal{A} \to \mathbb{R}^d$ such that $\mathbf{x}(\text{terminal}, \cdot) = \mathbf{0}$  
**Input:** a policy $\pi$ (if estimating $q_\pi$)  
**Algorithm parameters:** step size $\alpha > 0$, trace decay rate $\lambda \in [0, 1]$, small $\varepsilon > 0$  
**Initialize:** $\mathbf{w} \in \mathbb{R}^d$ (e.g., $\mathbf{w} = \mathbf{0}$)  

---

Loop for each episode:
* Initialize $S$
* Choose $A \sim \pi(\cdot|S)$ or $\varepsilon$-greedy according to $\hat{q}(S, \cdot, \mathbf{w})$
* $\mathbf{x} \leftarrow \mathbf{x}(S, A)$
* $\mathbf{z} \leftarrow \mathbf{0}$
* $Q_{\text{old}} \leftarrow 0$
* Loop for each step of episode:
  * | Take action $A$, observe $R, S'$
  * | Choose $A' \sim \pi(\cdot|S')$ or $\varepsilon$-greedy according to $\hat{q}(S', \cdot, \mathbf{w})$
  * | $\mathbf{x}' \leftarrow \mathbf{x}(S', A')$
  * | $Q \leftarrow \mathbf{w}^\top \mathbf{x}$
  * | $Q' \leftarrow \mathbf{w}^\top \mathbf{x}'$
  * | $\delta \leftarrow R + \gamma Q' - Q$
  * | $\mathbf{z} \leftarrow \gamma \lambda \mathbf{z} + (1 - \alpha \gamma \lambda \mathbf{z}^\top \mathbf{x})\mathbf{x}$
  * | $\mathbf{w} \leftarrow \mathbf{w} + \alpha(\delta + Q - Q_{\text{old}})\mathbf{z} - \alpha(Q - Q_{\text{old}})\mathbf{x}$
  * | $Q_{\text{old}} \leftarrow Q'$
  * | $\mathbf{x} \leftarrow \mathbf{x}'$
  * | $A \leftarrow A'$
* until $S'$ is terminal