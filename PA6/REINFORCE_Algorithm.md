# REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for $\pi_*$

**Input:** a differentiable policy parameterization $\pi(a|s, \boldsymbol{\theta})$  
**Algorithm parameter:** step size $\alpha > 0$  
**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Generate an episode $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot|\cdot, \boldsymbol{\theta})$
* **Loop for each step of the episode** $t = 0, 1, \dots, T - 1$:
    * $G \leftarrow \sum_{k=t+1}^T \gamma^{k-t-1} R_k \qquad \qquad \qquad$ $(G_t)$
    * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \gamma^t G \nabla \ln \pi(A_t|S_t, \boldsymbol{\theta})$

# REINFORCE with Baseline (episodic), for estimating $\pi_\theta \approx \pi_*$

**Input:** a differentiable policy parameterization $\pi(a|s, \boldsymbol{\theta})$  
**Input:** a differentiable state-value function parameterization $\hat{v}(s, \mathbf{w})$  
**Algorithm parameters:** step sizes $\alpha^{\boldsymbol{\theta}} > 0, \alpha^{\mathbf{w}} > 0$  
**Initialize** policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ and state-value weights $\mathbf{w} \in \mathbb{R}^d$ (e.g., to $\mathbf{0}$)

**Loop forever** (for each episode):
* Generate an episode $S_0, A_0, R_1, \dots, S_{T-1}, A_{T-1}, R_T$, following $\pi(\cdot|\cdot, \boldsymbol{\theta})$
* **Loop for each step of the episode** $t = 0, 1, \dots, T - 1$:
    * $G \leftarrow \sum_{k=t+1}^T \gamma^{k-t-1} R_k \qquad \qquad \qquad$ $(G_t)$
    * $\delta \leftarrow G - \hat{v}(S_t, \mathbf{w})$
    * $\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \nabla \hat{v}(S_t, \mathbf{w})$
    * $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}} \gamma^t \delta \nabla \ln \pi(A_t|S_t, \boldsymbol{\theta})$