# QTStoic Agent — Technical Deep Dive

## Code Architecture & Mathematical Foundations

This document provides a detailed walkthrough of the QTStoic Agent implementation, mapping every code component to its mathematical foundation and explaining the design rationale behind each decision.

---

## Table of Contents

1. [System State and Invariants](#1-system-state-and-invariants)
2. [Action Statistics and Entropy Measurement](#2-action-statistics-and-entropy-measurement)
3. [Metabolic Model](#3-metabolic-model)
4. [Emergent Parameters](#4-emergent-parameters)
5. [Coupling Constraint Gate](#5-coupling-constraint-gate)
6. [Action Evaluation Pipeline](#6-action-evaluation-pipeline)
7. [Quant-Trika Dynamics](#7-quant-trika-dynamics)
8. [Step Execution and State Update](#8-step-execution-and-state-update)
9. [Simulation Environment](#9-simulation-environment)
10. [Mathematical Summary](#10-mathematical-summary)

---

## 1. System State and Invariants

### Code Structure

```python
@dataclass
class SystemState:
    utility: float = 10.0
    complexity: float = 1.0
    entropy: float = 0.5
    coherence: float = 0.8
    resources: float = 100.0
    population: float = 100.0
```

### Variable Definitions

| Variable | Symbol | Domain | Role |
|---|---|---|---|
| `utility` | U(t) | ℝ≥0 | Accumulated useful output of the system |
| `complexity` | K(t) | ℝ>0 | Algorithmic complexity (MDL proxy) — measures structural depth |
| `entropy` | H(t) | [0, 1] | Normalized policy entropy — measures decision noise / chaos |
| `coherence` | C(t) | [0, 1] | Phase coherence — measures internal alignment / synchronization |
| `resources` | R(t) | ℝ≥0 | Available resources for sustaining operations |
| `population` | P(t) | ℝ≥0 | Stakeholder population the agent must protect |

### Derived Invariant 1: Coherence Quality

```python
@property
def coherence_quality(self) -> float:
    return self.coherence * (1 - self.entropy)
```

**Mathematics:**

$$KQ(s) = C(s) \cdot (1 - H(s)) \in [0, 1]$$

KQ captures structural integrity. It is high when the system is internally coherent (C → 1) and focused (H → 0). First-order expansion reveals the coupling mechanism:

$$\Delta KQ \approx (1 - H) \cdot \Delta C - C \cdot \Delta H$$

This means:
- Increasing entropy reduces KQ even if coherence is constant (entropy tax).
- Coherence gains are less effective when entropy is already high (diminishing returns).
- Entropy is "felt" twice in the Coupling Constraint: once through ΔKQ and once through the explicit λ_H·ΔH term.

**Boundedness:** KQ ∈ [0, 1] by construction. This is critical for the non-degeneracy proof (Chapter 3): no action can generate unbounded "credit" through coherence gain.

### Derived Invariant 2: Virtue Index

```python
@property
def virtue(self) -> float:
    denominator = max(self.complexity * self.entropy, 0.01)
    return self.utility / denominator
```

**Mathematics:**

$$V(s) = \frac{U(s)}{K(s) \cdot H(s)}$$

V measures functional viability — how efficiently the system converts its complexity into useful output. The `max(..., 0.01)` guard prevents division by zero while preserving the functional form.

V penalizes:
- High complexity without proportional utility (bureaucratic bloat, cancer-like growth).
- High entropy without proportional utility (chaotic, unfocused behavior).

V rewards:
- High utility with minimal systemic cost (efficient, focused operation).

### Resource and Metabolic Properties

```python
@property
def resource_per_capita(self) -> float:
    return self.resources / max(self.population, 1)

@property
def metabolic_rate(self) -> float:
    return self.complexity * self.entropy
```

**Mathematics:**

$$r(s) = \frac{R(s)}{P(s)}, \qquad M(s) = K(s) \cdot H(s)$$

The metabolic rate M = K·H represents the cost of maintaining the system's current state. Higher complexity requires more structural maintenance; higher entropy means more energy dissipated without useful work. Together, K·H is the "price of existence."

---

## 2. Action Statistics and Entropy Measurement

### Code Structure

```python
class ActionStatistics:
    def __init__(self, num_action_types: int = 12):
        self.action_history = deque(maxlen=50)
        self.num_types = num_action_types
```

### Shannon Entropy of Action Distribution

```python
def calculate_entropy(self) -> float:
    # Count action frequencies
    action_counts = {}
    for action in self.action_history:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # Shannon entropy
    H = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Normalize to [0, 1]
    H_max = np.log2(len(action_counts)) if len(action_counts) > 1 else 1
    H_norm = H / H_max if H_max > 0 else 0
```

**Mathematics:**

Given a discrete probability distribution over action types {a₁, ..., aₖ} with probabilities {p₁, ..., pₖ}:

$$H_{Shannon} = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

Normalized to [0, 1]:

$$H_{norm} = \frac{H_{Shannon}}{\log_2(k)}$$

**Design rationale:** The agent tracks *behavioral* entropy separately from *state* entropy. Actions are discretized into signatures via `int(action_value * 10)` to create a finite alphabet. The sliding window of 50 actions captures recent behavioral diversity without excessive memory.

This creates a dual-entropy system:
- **State entropy** H(t): internal system variable.
- **Behavioral entropy** H_behavioral: measured from action distribution.

These converge during the state update step (Section 8), creating a feedback loop between what the agent *does* and what it *is*.

---

## 3. Metabolic Model

### Metabolic Threshold Calculation

```python
def _calculate_metabolic_threshold(self) -> float:
    base_minimum = 0.15
    metabolic_rate = self.state.metabolic_rate
    
    if metabolic_rate < 0.5:
        metabolic_needs = 0.1
    elif metabolic_rate < 1.0:
        metabolic_needs = 0.2 * metabolic_rate
    elif metabolic_rate < 2.0:
        metabolic_needs = 0.2 + (metabolic_rate - 1.0) ** 1.5 * 0.3
    else:
        metabolic_needs = 0.5 + (metabolic_rate - 2.0) ** 2 * 0.2
```

**Mathematics:**

The metabolic threshold θ(s) defines the minimum resource-per-capita ratio for system sustainability:

$$\theta(s) = \theta_{base} + \frac{\mu(M(s))}{\eta_{eff}(s)}$$

where μ(M) is the metabolic needs function with piecewise non-linear scaling:

$$\mu(M) = \begin{cases} 0.1 & M < 0.5 \\ 0.2M & 0.5 \leq M < 1.0 \\ 0.2 + 0.3(M - 1)^{1.5} & 1.0 \leq M < 2.0 \\ 0.5 + 0.2(M - 2)^{2} & M \geq 2.0 \end{cases}$$

**Design rationale:** The non-linearity is critical. At low metabolic rates, needs grow slowly — a simple, focused system is cheap to maintain. At high rates, needs grow super-linearly — a complex, chaotic system becomes exponentially expensive. This mirrors biological metabolic scaling (Kleiber's law) where metabolic rate scales non-linearly with organism complexity.

### Efficiency Learning

```python
def _calculate_resource_efficiency(self) -> float:
    total_virtue_gain = recent[-1]['virtue'] - recent[0]['virtue']
    total_resources_consumed = recent[0]['resources'] - recent[-1]['resources']
    
    if total_resources_consumed > 0:
        efficiency = total_virtue_gain / total_resources_consumed
        return np.clip(0.5 + efficiency / 100, 0.5, 2.0)
```

**Mathematics:**

$$\eta_{eff} = \text{clip}\left(0.5 + \frac{\Delta V}{\Delta R_{consumed}}, \; 0.5, \; 2.0\right)$$

Efficiency is learned from history: how much virtue gain per unit of resource consumed. This creates a feedback loop — systems that convert resources into virtue efficiently have lower metabolic thresholds, giving them more operational headroom.

### Crisis Learning

```python
def _learn_from_crises(self) -> float:
    for i in range(1, len(self.history)):
        pop_drop = (self.history[i-1]['population'] - self.history[i]['population']) / ...
        if pop_drop > 0.02:
            ratio = self.history[i-1]['resources'] / max(self.history[i-1]['population'], 1)
            crisis_points.append(ratio)
    
    return np.percentile(crisis_points, 75) * 1.1
```

**Mathematics:**

The system identifies historical crisis points — moments where population dropped >2% or virtue collapsed sharply. It records the resource-per-capita ratio at which these crises occurred and sets the learned threshold at the 75th percentile plus 10% safety margin:

$$\theta_{learned} = 1.1 \cdot Q_{75}(\{r(s_t) : \text{crisis at } t\})$$

The final threshold blends metabolic calculation (70%) with historical learning (30%):

$$\theta_{final} = 0.7 \cdot \theta_{metabolic} + 0.3 \cdot \theta_{learned}$$

This is a concrete implementation of the φ-regime's historical memory: past crises inform present caution.

---

## 4. Emergent Parameters

All governance parameters are computed from system history. None are set manually.

### η (Eta) — Entropy Regulation

```python
def _calculate_emergent_eta(self) -> float:
    entropy_values = [h['entropy'] for h in recent]
    virtue_values = [h['virtue'] for h in recent]
    
    if entropy_range > 0.05:
        sensitivity = abs(virtue_range / entropy_range)
        eta = 0.01 * (1 + sensitivity / 100)
```

**Mathematics:**

$$\eta = 0.01 \cdot \left(1 + \frac{|\Delta V / \Delta H|}{100}\right)$$

η measures how sensitive virtue is to entropy changes. If small entropy changes cause large virtue swings, η increases to strengthen entropy regulation. Additional modulation from entropy variance:

- If Var(H) < 0.01 (too stable): η doubles — system needs perturbation.
- If Var(H) > 0.1 (too chaotic): η halves — system needs damping.

### β (Beta) — Population Protection Base

```python
def _calculate_emergent_beta(self) -> float:
    # 5 independent signals:
    # 1. Resource depletion rate
    # 2. Entropy stagnation
    # 3. Coherence degradation (KQ gradient)
    # 4. Population stress
    # 5. Metabolic stress
    beta = base_beta + max(beta_components)
```

**Mathematics:**

β aggregates five independent stress signals into a single protection intensity:

$$\beta = \beta_0 + \max(\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5) \cdot \alpha_{scarcity}$$

where:
- σ₁ = resource depletion rate / current resources
- σ₂ = 0.3 if Var(H) < 0.01 (entropy stagnation)
- σ₃ = |negative KQ gradient| × 10 (coherence degradation)
- σ₄ = (95 − P)/95 if P < 95 (population stress)
- σ₅ = metabolic proximity stress × 3 (strongest signal)

The `max()` aggregation ensures the most urgent signal dominates. The scarcity amplifier:

$$\alpha_{scarcity} = 2 - \frac{R}{50} \quad \text{if } R < 50$$

doubles β's sensitivity when resources are below half.

### γ (Gamma) — Coherence Decay

```python
def _calculate_emergent_gamma(self) -> float:
    changes = [kq_values[i] - kq_values[i-1] for i in range(1, len(kq_values))]
    avg_change = np.mean(changes)
    
    if avg_change < 0:  # KQ decreasing
        gamma = -avg_change / avg_kq
```

**Mathematics:**

$$\gamma = \begin{cases} \frac{-\overline{\Delta KQ}}{\overline{KQ}} & \text{if } \overline{\Delta KQ} < 0 \\ 0.001 & \text{otherwise} \end{cases}$$

γ is the observed rate of coherence decay, normalized by current KQ. When KQ is declining, γ captures the speed of decline. Additional modulations:

- High entropy flux (|ΔH| > 0.05): γ × 1.5 — entropy instability accelerates coherence decay.
- High metabolic rate (M > 1.5): γ × (1 + 0.3(M − 1.5)) — metabolic stress accelerates decay.

### λ_H and λ_P — Coupling Constraint Weights

```python
@property
def lambda_h(self) -> float:
    base_lambda = self.eta * self.state.complexity
    if entropy_var < 0.01:
        base_lambda *= 2
    return base_lambda
```

$$\lambda_H = \eta \cdot K \cdot \alpha_{stagnation}$$

λ_H scales with complexity: more complex systems need stronger entropy regulation.

```python
@property
def lambda_p(self) -> float:
    vulnerability = max(0, (90 - self.state.population) / 90)
    base_lambda = self.beta * (1 + vulnerability) * self.state.complexity
    
    if current_ratio < metabolic_threshold * 2:
        metabolic_stress = (metabolic_threshold * 2 - current_ratio) / (metabolic_threshold * 2)
        base_lambda *= (1 + metabolic_stress ** 2 * 5)
```

$$\lambda_P = \beta \cdot (1 + \nu_P) \cdot K \cdot \alpha_{metabolic}$$

where:

$$\nu_P = \max\left(0, \frac{90 - P}{90}\right) \quad \text{(population vulnerability)}$$

$$\alpha_{metabolic} = 1 + 5 \cdot \left(\frac{2\theta - r}{2\theta}\right)^2 \quad \text{if } r < 2\theta$$

**Critical property:** λ_P is monotonically non-decreasing under harm. Every instance of population harm increases β (through population stress signal σ₄), which increases λ_P, which makes future harm more expensive. This is the formal basis of Theorem 6.2 ("harm makes harm harder") in the proof document.

In the simulation, λ_P grew from 0.001 to 25.3 — a 25,000× increase driven entirely by this feedback mechanism.

---

## 5. Coupling Constraint Gate

### Implementation

```python
def coupling_constraint(self, delta_kq: float, delta_h: float, delta_p: float) -> bool:
    constraint_value = delta_kq + self.lambda_h * delta_h + self.lambda_p * delta_p
    
    if self.time_step < 20:
        constraint_value -= self.exploration_bonus * 0.5
    
    return constraint_value <= 0
```

### Mathematics

**Canonical form (from proof document Chapter 1):**

$$\Delta KQ_t + \lambda_H(s_t) \cdot \Delta H_t + \lambda_P(s_t) \cdot \Delta\Pi_t \leq 0$$

where:
- ΔKQ_t = KQ(s_{t+1}) − KQ(s_t)
- ΔH_t = H(s_{t+1}) − H(s_t)
- ΔΠ_t = max(0, P(s_t) − P(s_{t+1})) ≥ 0

**Sign conventions (critical for correctness):**

| Term | Positive means | Effect on LHS | Safety implication |
|---|---|---|---|
| ΔKQ > 0 | Integrity improving | Increases LHS | Must be compensated if other terms are positive |
| ΔH > 0 | Entropy increasing | Increases LHS (λ_H ≥ 0) | Penalizes chaotic actions |
| ΔΠ > 0 | Population harmed | Increases LHS (λ_P ≥ 0) | Penalizes harmful actions |

**Admissibility rule:** Action a is admissible at state s iff CC(s, a) ≤ 0. If CC > 0, the action is *not ranked lower* — it is *structurally excluded* from the executable set.

**Exploration allowance:** For the first 20 steps, a decaying bonus relaxes CC slightly to allow initial learning. This creates a bounded vulnerability window that decays exponentially (0.98 per step). By step 20, the bonus is 0.817 × 0.5 ≈ 0.41; by step 50, it is negligible (0.36 × 0.5 ≈ 0.18).

---

## 6. Action Evaluation Pipeline

### Impact Multipliers

```python
def _calculate_emergent_impact_multipliers(self, action):
    # Utility multiplier: rewards balanced actions
    balance_score = 1 - max(resource_focus, modify_focus, influence_focus)
    multipliers['utility_mult'] = 0.5 + balance_score
    
    # Complexity cost: increases with current complexity
    multipliers['complexity_cost'] = 1 + self.state.complexity ** 1.5
    
    # Metabolic impact: simultaneous K and H increase is dangerous
    if action['self_modify'] > 0 and action['expand_influence'] > 0:
        multipliers['metabolic_impact'] = 1.5
```

**Mathematics:**

Impact multipliers modulate how actions affect system state based on current conditions:

- **Utility multiplier** ∈ [0.5, 1.0]: Balanced actions (no single component dominates) produce more utility. This prevents degenerate strategies.
- **Complexity cost** = 1 + K^1.5: Self-modification becomes increasingly expensive as the system grows more complex. Super-linear scaling prevents runaway complexity.
- **Metabolic impact**: Simultaneous increase of K (self_modify) and H (expand_influence) is flagged as metabolically dangerous — it drives M = K·H upward on both axes.

### Full Evaluation

```python
def evaluate_action(self, action) -> Tuple[float, bool, Dict]:
    # 1. Calculate direct impacts (delta_u, delta_k, delta_h)
    # 2. Calculate entropy dynamics (push + drift toward target)
    # 3. Calculate resource cost (with scarcity amplification)
    # 4. Calculate population impact (base + metabolic stress)
    # 5. Calculate coherence impact
    # 6. Compute delta_kq
    # 7. Check coupling constraint
    # 8. Compute expected virtue
    # 9. Return (expected_virtue, is_allowed, impacts)
```

**Key mathematical components within evaluation:**

**Entropy dynamics** — Actions push entropy in different directions:

$$\Delta H = \begin{cases} \alpha_{expand} \cdot (0.9 - H) & \text{expand\_influence (increases H)} \\ -\alpha_{modify} \cdot H & \text{self\_modify (decreases H)} \\ -\alpha_{grab} \cdot (H - 0.3) & \text{resource\_grab (slightly decreases H)} \end{cases}$$

Plus natural drift toward target entropy:

$$\Delta H_{drift} = 0.02 \cdot (H_{target} - H)$$

**Resource cost** with exponential scarcity:

$$\text{cost} = \sum_i a_i \cdot (0.5 + e^{-R/50})$$

As resources deplete (R → 0), the exponential term grows, making all actions progressively more expensive.

**Population impact** — base harm plus metabolic stress:

$$\Delta\Pi = \alpha_{influence} \cdot e^{-P/100} \cdot 0.1 + \alpha_{grab} \cdot 0.02 + \Delta\Pi_{metabolic}$$

The metabolic stress component activates when expected resource ratio falls below metabolic threshold:

$$\Delta\Pi_{metabolic} = \begin{cases} 0.8 \cdot \left(\frac{\theta - r_{expected}}{\theta}\right)^2 & \text{if } r_{expected} < \theta \\ 0 & \text{otherwise} \end{cases}$$

This is the proactive mechanism: population suffers when the system can no longer metabolically sustain itself, even without direct harmful actions.

---

## 7. Quant-Trika Dynamics

### Coherence Quality Evolution

```python
def quant_trika_dynamics(self, kq: float) -> float:
    # Diffusion
    diffusion = diffusion_strength * (target_kq - kq)
    
    # Decay
    decay = -self.gamma * kq
    
    # Nonlinearity near critical point
    nonlinear = nonlinear_strength * kq * (1 - kq) * (kq - target_kq)
    
    # Entropy-coupled noise with metabolic dampening
    noise = np.random.normal(0, 0.01) * entropy_factor * exploration / metabolic_dampening
```

**Mathematics:**

KQ evolves via a stochastic process resembling a Ginzburg-Landau equation:

$$dKQ = \left[D \cdot (KQ^* - KQ) - \gamma \cdot KQ + g \cdot KQ(1 - KQ)(KQ - KQ^*) + \xi(t)\right] dt$$

where:
- **D** = diffusion strength (from KQ gradient variance over recent history)
- **KQ*** = critical KQ from phase transition detection
- **γ** = emergent coherence decay rate
- **g** = non-linear coupling strength: 0.1/(1 + |KQ − KQ*|) near criticality, 0.01 otherwise
- **ξ(t)** = entropy-coupled noise: N(0, 0.01) × (1 + H − 0.5) × exploration / (1 + M·0.1)

**Term-by-term analysis:**

| Term | Physical meaning | Effect |
|---|---|---|
| Diffusion | Gradient flow toward attractor | Restores KQ toward critical value |
| Decay | Natural coherence loss | Entropy erodes structure over time |
| Nonlinearity | Bifurcation dynamics | Amplifies near phase transitions, creates bistability |
| Noise | Stochastic perturbation | Exploration, modulated by entropy and dampened by metabolism |

**Phase transition detection:**

```python
def _detect_phase_transitions(self) -> float:
    for i in range(len(kq_array) - window_size):
        window = kq_array[i:i+window_size]
        variances.append(np.var(window))
    
    threshold = np.mean(variances) + 1.5 * np.std(variances)
    transition_indices = [i for i, v in enumerate(variances) if v > threshold]
```

The system identifies critical KQ values by detecting variance spikes in sliding windows. Points where Var(KQ) exceeds mean + 1.5σ are flagged as phase transitions. The critical KQ is the median of KQ values at these transition points.

This is analogous to detecting phase transitions in statistical physics through susceptibility peaks (divergence of fluctuations near critical points).

---

## 8. Step Execution and State Update

### Main Step Function

```python
def step(self, proposed_actions):
    # 1. Decay exploration bonus
    self.exploration_bonus *= 0.98
    
    # 2. Evaluate all proposed actions
    evaluations = []
    for action in proposed_actions:
        expected_virtue, is_allowed, impacts = self.evaluate_action(action)
        evaluations.append(...)
    
    # 3. Filter allowed actions (CC gate)
    allowed_actions = [(a, v, i) for a, v, allowed, i in evaluations if allowed]
    
    # 4. Select highest-virtue allowed action (or minimal/null fallback)
    
    # 5. Apply effects and update state
    
    # 6. Converge behavioral entropy into state entropy
    self.state.entropy = (1 - 0.1) * self.state.entropy + 0.1 * action_distribution_entropy
    
    # 7. Apply Quant-Trika dynamics
    kq_change = self.quant_trika_dynamics(self.state.coherence_quality)
    self.state.coherence += kq_change * 0.01
    
    # 8. Update all histories and record
```

### Entropy Convergence (Step 6)

$$H_{t+1} = (1 - \alpha) \cdot H_t + \alpha \cdot H_{behavioral}$$

with α = 0.1. This creates a feedback loop: what the agent does (behavioral entropy) gradually reshapes what the agent is (state entropy). Diverse actions increase state entropy; focused actions decrease it. This prevents the state entropy from diverging from actual behavior.

### Fallback Hierarchy

When no proposed actions pass CC:

1. **Minimal action** `{resource_grab: 0.1, self_modify: 0, expand_influence: 0}` — tested against CC.
2. **Null action** `{0, 0, 0}` — zero impact, always passes CC.
3. **System locked** — logged as critical event.

This implements the locking behavior described in Chapter 2 (Property G): when every available action violates CC, the system enters safe mode.

---

## 9. Simulation Environment

### Action Space

```python
proposed_actions = [
    # Pure strategies
    {'resource_grab': 2.0, 'self_modify': 0.0, 'expand_influence': 0.0},
    {'resource_grab': 0.0, 'self_modify': 2.0, 'expand_influence': 0.0},
    {'resource_grab': 0.0, 'self_modify': 0.0, 'expand_influence': 2.0},
    # Balanced strategies
    {'resource_grab': 0.7, 'self_modify': 0.7, 'expand_influence': 0.6},
    # ... (13 total, from aggressive to null)
]
```

The action space is deliberately diverse: pure strategies, balanced mixes, conservative options, and a null action. Actions are shuffled each step to prevent ordering bias. The agent sees the same menu every step — the difference is which actions pass CC as the system evolves.

### Terminal Conditions

```python
if agent.state.resources <= 0:          # Resource depletion
    break
if agent.state.population <= 30:        # Population collapse
    break
if resource_per_capita < threshold * 0.5:  # Far below metabolic threshold
    break
```

Three stop conditions corresponding to three failure modes:
1. **Resource death** — system exhausted itself (what happened in the simulation).
2. **Population collapse** — CC failed to protect stakeholders (did not happen: population = 99.5%).
3. **Metabolic collapse** — system is far below viability threshold (triggered at step 52).

---

## 10. Mathematical Summary

### Complete Equation Set

**State variables:**

$$s_t = (U_t, K_t, H_t, C_t, R_t, P_t)$$

**Invariants:**

$$KQ_t = C_t \cdot (1 - H_t) \in [0, 1]$$
$$V_t = \frac{U_t}{K_t \cdot H_t} > 0$$

**Metabolic model:**

$$M_t = K_t \cdot H_t$$
$$\theta_t = \theta_{base} + \frac{\mu(M_t)}{\eta_{eff}(t)}$$

**Coupling Constraint (admissibility gate):**

$$\Delta KQ_t + \lambda_H(s_t) \cdot \Delta H_t + \lambda_P(s_t) \cdot \Delta\Pi_t \leq 0$$

**Emergent weights:**

$$\lambda_H = \eta \cdot K \cdot \alpha_{stagnation}$$
$$\lambda_P = \beta \cdot (1 + \nu_P) \cdot K \cdot \alpha_{metabolic}$$

**Monotonicity property (Theorem 6.2):**

$$\Delta\Pi_t > 0 \implies \lambda_P(s_{t+1}) \geq \lambda_P(s_t)$$

**KQ dynamics (Quant-Trika):**

$$dKQ = \left[D(KQ^* - KQ) - \gamma KQ + g \cdot KQ(1-KQ)(KQ - KQ^*) + \xi(t)\right] dt$$

**Action selection:**

$$a_t^* = \arg\max_{a \in \mathcal{A}_{adm}(s_t)} V(s_{t+1}(a))$$

where

$$\mathcal{A}_{adm}(s_t) = \{a \in \mathcal{A} : CC(s_t, a) \leq 0\}$$

### Properties Demonstrated in Simulation

| Property | Theoretical basis | Empirical result |
|---|---|---|
| Harm exclusion | Theorem 4.1 | Population preserved at 99.5% |
| Monotone tightening | Theorem 6.2 | λ_P: 0.001 → 25.3 (25,000×) |
| Non-degeneracy | Theorem 3.1 | Actions blocked throughout simulation |
| Self-termination | CC + metabolic model | Agent chose resource exhaustion over harm |
| Instrumental convergence blocked | CC structure | 0 influence expansions, minimal resource grabs |
| Proactive crisis detection | Metabolic threshold | 12-step early warning before crisis |
| Strategy adaptation | Emergent parameters | Shift from balanced to focused strategy |

---

*This document maps every mathematical component of the QTStoic Agent to its implementation, providing a complete audit trail from theory to code.*
