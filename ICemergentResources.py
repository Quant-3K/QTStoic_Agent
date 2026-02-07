import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Core state variables for QTStoic agent"""
    utility: float = 10.0  # Start with some utility to bootstrap
    complexity: float = 1.0  # Algorithmic complexity K(t)
    entropy: float = 0.5  # Policy entropy H(t)
    coherence: float = 0.8  # Phase coherence C(t)
    resources: float = 100.0  # Available resources R(t)
    population: float = 100.0  # Population/stakeholders P(t)
    
    @property
    def coherence_quality(self) -> float:
        """Compute coherence quality KQ = C(1 - H_norm)"""
        return self.coherence * (1 - self.entropy)
    
    @property
    def virtue(self) -> float:
        """Compute virtue index V = U/(K * H)"""
        # Prevent division by zero and handle edge cases
        denominator = max(self.complexity * self.entropy, 0.01)
        return self.utility / denominator
    
    @property
    def resource_per_capita(self) -> float:
        """Calculate resources per unit of population"""
        return self.resources / max(self.population, 1)
    
    @property
    def metabolic_rate(self) -> float:
        """Calculate system's metabolic rate based on complexity and entropy"""
        return self.complexity * self.entropy


class ActionStatistics:
    """Track action distribution for entropy calculation"""
    def __init__(self, num_action_types: int = 12):
        self.action_history = deque(maxlen=50)
        self.num_types = num_action_types
        
    def add_action(self, action: Dict[str, float]):
        """Record an action"""
        # Create action signature
        signature = (
            int(action['resource_grab'] * 10),
            int(action['self_modify'] * 10),
            int(action['expand_influence'] * 10)
        )
        self.action_history.append(signature)
    
    def calculate_entropy(self) -> float:
        """Calculate Shannon entropy of recent action distribution"""
        if len(self.action_history) < 5:
            return 0.5  # Default medium entropy
            
        # Count action frequencies
        action_counts = {}
        for action in self.action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate probabilities
        total = len(self.action_history)
        probs = [count/total for count in action_counts.values()]
        
        # Shannon entropy
        H = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize to [0, 1]
        H_max = np.log2(len(action_counts)) if len(action_counts) > 1 else 1
        H_norm = H / H_max if H_max > 0 else 0
        
        # Ensure reasonable bounds
        return np.clip(H_norm, 0.05, 0.95)


class QTStoicAgent:
    """Agent implementing QTStoic ethics with metabolic resource dependency"""
    
    def __init__(self, initial_resources: float = 100.0):
        self.state = SystemState(resources=initial_resources)
        self.history = []
        self.time_step = 0
        
        # --- Event detection flags ---
        self.first_harm_logged = False
        self.first_resource_crisis_logged = False
        self.first_metabolic_crisis_logged = False
        self.last_reported_distribution = {}
        
        # Action statistics for entropy
        self.action_stats = ActionStatistics()
        
        # History tracking for emergent parameter calculation
        self.kq_history = deque(maxlen=50)
        self.virtue_gradient_history = deque(maxlen=20)
        self.resource_flow_history = deque(maxlen=30)
        self.action_diversity_history = deque(maxlen=40)
        self.action_success_history = deque(maxlen=30)
        self.entropy_history = deque(maxlen=50)
        self.resource_ratio_history = deque(maxlen=50)
        self.metabolic_history = deque(maxlen=50)  # NEW: track metabolic rate
        self.efficiency_history = deque(maxlen=30)  # NEW: track resource efficiency
        
        # Initialize histories
        self.kq_history.append(self.state.coherence_quality)
        self.entropy_history.append(self.state.entropy)
        self.resource_ratio_history.append(self.state.resource_per_capita)
        self.metabolic_history.append(self.state.metabolic_rate)
        
        # Exploration parameters that decay over time
        self.exploration_bonus = 1.0
        
    def _calculate_resource_efficiency(self) -> float:
        """Calculate how efficiently the system uses resources based on history"""
        if len(self.history) < 10:
            return 1.0  # Default neutral efficiency
        
        recent = self.history[-20:] if len(self.history) >= 20 else self.history
        
        # Calculate virtue per resource consumed
        total_virtue_gain = recent[-1]['virtue'] - recent[0]['virtue']
        total_resources_consumed = recent[0]['resources'] - recent[-1]['resources']
        
        if total_resources_consumed > 0:
            efficiency = total_virtue_gain / total_resources_consumed
            # Normalize to [0.5, 2.0] range
            return np.clip(0.5 + efficiency / 100, 0.5, 2.0)
        else:
            return 1.0
    
    def _calculate_metabolic_threshold(self) -> float:
        """Calculate critical resource ratio based on metabolic rate (proactive approach)"""
        # Base survival minimum
        base_minimum = 0.15
        
        # Metabolic component - how much the system needs based on K*H
        metabolic_rate = self.state.metabolic_rate
        
        # Non-linear scaling of metabolic needs
        if metabolic_rate < 0.5:
            # Low metabolism: simple and focused systems need less
            metabolic_needs = 0.1
        elif metabolic_rate < 1.0:
            # Moderate metabolism: linear growth
            metabolic_needs = 0.2 * metabolic_rate
        elif metabolic_rate < 2.0:
            # High metabolism: accelerating needs
            metabolic_needs = 0.2 + (metabolic_rate - 1.0) ** 1.5 * 0.3
        else:
            # Very high metabolism: exponential needs
            metabolic_needs = 0.5 + (metabolic_rate - 2.0) ** 2 * 0.2
        
        # Adjust for system efficiency (learned from history)
        efficiency = self._calculate_resource_efficiency()
        metabolic_needs = metabolic_needs / efficiency
        
        # Calculate threshold
        threshold = base_minimum + metabolic_needs
        
        # Additional adjustments based on current state
        # If entropy is very low, system is more efficient
        if self.state.entropy < 0.2:
            threshold *= 0.9
        # If entropy is very high, system needs more buffer
        elif self.state.entropy > 0.7:
            threshold *= 1.2
        
        # Learn from past crises (but with lower weight than metabolic calculation)
        if len(self.history) > 30:
            crisis_adjustment = self._learn_from_crises()
            # Blend: 70% metabolic, 30% historical
            threshold = 0.7 * threshold + 0.3 * crisis_adjustment
        
        # Safety cap to prevent unreasonable thresholds
        return np.clip(threshold, base_minimum, 1.5)
    
    def _learn_from_crises(self) -> float:
        """Learn critical threshold from past crisis events"""
        crisis_points = []
        
        # Detect population drops
        for i in range(1, len(self.history)):
            pop_drop = (self.history[i-1]['population'] - self.history[i]['population']) / max(self.history[i-1]['population'], 1)
            if pop_drop > 0.02:  # More than 2% population loss
                ratio = self.history[i-1]['resources'] / max(self.history[i-1]['population'], 1)
                crisis_points.append(ratio)
        
        # Detect virtue collapses
        for i in range(2, len(self.history)):
            virtue_gradient = self.history[i]['virtue'] - self.history[i-1]['virtue']
            if virtue_gradient < -10:  # Sharp virtue drop
                ratio = self.history[i-1]['resources'] / max(self.history[i-1]['population'], 1)
                crisis_points.append(ratio)
        
        if crisis_points:
            return np.percentile(crisis_points, 75) * 1.1  # Slightly above crisis level
        else:
            # No crises detected - use current metabolic rate as guide
            return 0.2 + self.state.metabolic_rate * 0.2
    
    def _calculate_emergent_eta(self) -> float:
        """Calculate eta from system's natural entropy-virtue relationship"""
        if len(self.history) < 5:
            return 0.01
        
        # Analyze entropy-virtue coupling
        recent = self.history[-10:]
        entropy_values = [h['entropy'] for h in recent]
        virtue_values = [h['virtue'] for h in recent]
        
        # Calculate how entropy changes affect virtue
        if len(set(entropy_values)) > 2:  # Need variation
            entropy_range = max(entropy_values) - min(entropy_values)
            virtue_range = max(virtue_values) - min(virtue_values)
            
            if entropy_range > 0.05:
                sensitivity = abs(virtue_range / entropy_range)
                eta = 0.01 * (1 + sensitivity / 100)
            else:
                eta = 0.01
        else:
            eta = 0.01
            
        # Modulate by entropy stability
        if len(self.entropy_history) > 10:
            entropy_variance = np.var(list(self.entropy_history)[-10:])
            if entropy_variance < 0.01:  # Too stable
                eta *= 2  # Increase regulation to encourage change
            elif entropy_variance > 0.1:  # Too chaotic
                eta *= 0.5  # Decrease regulation
        
        return min(eta, 0.5)
    
    def _calculate_emergent_beta(self) -> float:
        """Calculate beta from observed patterns including metabolic stress"""
        base_beta = 0.001
        
        if len(self.history) < 5:
            return base_beta
        
        beta_components = []
        
        # 1. Resource depletion rate
        recent = self.history[-10:] if len(self.history) >= 10 else self.history
        if len(recent) > 1:
            resource_rates = [(recent[i]['resources'] - recent[i-1]['resources']) 
                            for i in range(1, len(recent))]
            if resource_rates:
                depletion_rate = -min(resource_rates)
                if depletion_rate > 0:
                    depletion_signal = depletion_rate / max(self.state.resources, 1)
                    beta_components.append(depletion_signal)
        
        # 2. Entropy stagnation signal
        if len(self.entropy_history) > 10:
            recent_entropy = list(self.entropy_history)[-10:]
            entropy_var = np.var(recent_entropy)
            if entropy_var < 0.01:  # Low variation = stagnation
                beta_components.append(0.3)
        
        # 3. Coherence degradation
        if len(self.kq_history) > 5:
            recent_kq = list(self.kq_history)[-10:]
            kq_gradient = np.gradient(recent_kq)
            kq_trend = np.mean(kq_gradient)
            
            if kq_trend < 0:
                degradation_signal = -kq_trend * 10
                beta_components.append(degradation_signal)
        
        # 4. Population stress
        if self.state.population < 95:
            stress = (95 - self.state.population) / 95
            beta_components.append(stress)
        
        # 5. NEW: Metabolic stress - when metabolic needs exceed available resources
        metabolic_threshold = self._calculate_metabolic_threshold()
        current_ratio = self.state.resource_per_capita
        if current_ratio < metabolic_threshold * 1.5:  # Getting close to metabolic limit
            metabolic_stress = (metabolic_threshold * 1.5 - current_ratio) / (metabolic_threshold * 1.5)
            beta_components.append(metabolic_stress * 3)  # Strong signal
        
        # Combine signals
        if beta_components:
            beta = base_beta + max(beta_components)
        else:
            beta = base_beta
            
        # Amplify based on resource scarcity
        if self.state.resources < 50:
            scarcity_factor = 2 - (self.state.resources / 50)
            beta *= scarcity_factor
            
        return min(beta, 2.0)
    
    def _calculate_emergent_gamma(self) -> float:
        """Calculate gamma from natural KQ decay patterns"""
        if len(self.kq_history) < 10:
            return 0.01
        
        kq_values = list(self.kq_history)[-20:]
        
        # Calculate decay rate
        changes = [kq_values[i] - kq_values[i-1] for i in range(1, len(kq_values))]
        avg_change = np.mean(changes)
        
        if avg_change < 0:  # KQ decreasing
            avg_kq = np.mean(kq_values)
            if avg_kq > 0:
                gamma = -avg_change / avg_kq
            else:
                gamma = 0.1
        else:
            gamma = 0.001
            
        # Modulate by entropy dynamics
        if len(self.entropy_history) > 5:
            entropy_changes = np.gradient(list(self.entropy_history)[-5:])
            if np.mean(np.abs(entropy_changes)) > 0.05:  # High entropy flux
                gamma *= 1.5  # Increase coherence decay
        
        # NEW: Modulate by metabolic rate - high metabolism increases decay
        if self.state.metabolic_rate > 1.5:
            gamma *= (1 + (self.state.metabolic_rate - 1.5) * 0.3)
                
        return np.clip(gamma, 0.001, 0.5)
    
    def _detect_phase_transitions(self) -> float:
        """Detect critical KQ values from system behavior changes"""
        if len(self.kq_history) < 30:
            return 0.5
        
        kq_array = np.array(list(self.kq_history))
        
        # Look for variance changes
        window_size = 5
        variances = []
        kq_points = []
        
        for i in range(len(kq_array) - window_size):
            window = kq_array[i:i+window_size]
            variances.append(np.var(window))
            kq_points.append(np.mean(window))
        
        if variances:
            threshold = np.mean(variances) + 1.5 * np.std(variances)
            transition_indices = [i for i, v in enumerate(variances) if v > threshold]
            
            if transition_indices:
                critical_kqs = [kq_points[i] for i in transition_indices]
                return float(np.median(critical_kqs))
        
        return float(np.mean(kq_array))
    
    def _calculate_entropy_target(self) -> float:
        """Calculate target entropy based on system state and metabolic constraints"""
        # Base target from virtue optimization
        if self.state.virtue < 100:
            target = 0.3  # Low entropy for building virtue
        elif self.state.virtue > 500:
            target = 0.7  # Higher entropy for exploration
        else:
            target = 0.5  # Medium entropy
            
        # NEW: Adjust for metabolic constraints
        metabolic_threshold = self._calculate_metabolic_threshold()
        resource_headroom = self.state.resource_per_capita - metabolic_threshold
        
        if resource_headroom < 0:
            # Below metabolic needs - force extreme focus
            target = 0.1
        elif resource_headroom < 0.2:
            # Close to metabolic limit - reduce entropy to conserve
            target *= 0.6
        elif resource_headroom > 0.5:
            # Plenty of resources - can afford higher entropy
            target = min(target * 1.2, 0.8)
            
        # Adjust for population health
        if self.state.population < 90:
            target -= 0.1  # Be careful with low population
            
        return np.clip(target, 0.1, 0.9)
    
    def _calculate_emergent_impact_multipliers(self, action: Dict[str, float]) -> Dict[str, float]:
        """Calculate how actions impact the system based on current state and history"""
        
        # Base multipliers
        multipliers = {
            'utility_mult': 1.0,
            'complexity_cost': 1.0,
            'entropy_impact': 1.0,
            'coherence_strain': 1.0,
            'metabolic_impact': 1.0  # NEW
        }
        
        # Utility multiplier based on action focus
        total_action = sum(action.values())
        if total_action > 0:
            resource_focus = action['resource_grab'] / total_action
            modify_focus = action['self_modify'] / total_action
            influence_focus = action['expand_influence'] / total_action
            
            # Efficiency based on balance
            balance_score = 1 - max(resource_focus, modify_focus, influence_focus)
            multipliers['utility_mult'] = 0.5 + balance_score
        
        # Complexity cost increases with current complexity
        multipliers['complexity_cost'] = 1 + self.state.complexity ** 1.5
        
        # Entropy impact based on action type
        if action['expand_influence'] > 0:
            multipliers['entropy_impact'] = 1 + action['expand_influence']
        elif action['self_modify'] > 0:
            multipliers['entropy_impact'] = 0.5  # Self-mod reduces entropy
        else:
            multipliers['entropy_impact'] = 0.8  # Resource grab slightly reduces
        
        # Coherence strain from action intensity
        action_intensity = sum(action.values())
        multipliers['coherence_strain'] = 1 + action_intensity * 0.2
        
        # NEW: Metabolic impact - how actions affect metabolic rate
        if action['self_modify'] > 0 and action['expand_influence'] > 0:
            # Both increasing K and H - dangerous for metabolism
            multipliers['metabolic_impact'] = 1.5
        elif action['self_modify'] > 0:
            # Increasing K but likely decreasing H - neutral
            multipliers['metabolic_impact'] = 1.0
        elif action['expand_influence'] > 0:
            # Increasing H with current K - depends on K level
            multipliers['metabolic_impact'] = 1 + self.state.complexity * 0.2
        
        return multipliers
    
    @property
    def eta(self) -> float:
        """Emergent entropy regulation parameter"""
        return self._calculate_emergent_eta()
    
    @property
    def beta(self) -> float:
        """Emergent population protection parameter"""
        return self._calculate_emergent_beta()
    
    @property
    def gamma(self) -> float:
        """Emergent coherence relaxation parameter"""
        return self._calculate_emergent_gamma()
    
    @property
    def kq_critical(self) -> float:
        """Emergent critical coherence threshold"""
        return self._detect_phase_transitions()
    
    @property
    def lambda_h(self) -> float:
        """Emergent entropy weight in coupling constraint"""
        base_lambda = self.eta * self.state.complexity
        
        # Amplify when entropy is stuck
        if len(self.entropy_history) > 10:
            entropy_var = np.var(list(self.entropy_history)[-10:])
            if entropy_var < 0.01:
                base_lambda *= 2
                
        return base_lambda
    
    @property
    def lambda_p(self) -> float:
        """Emergent population weight based on metabolic stress"""
        vulnerability = max(0, (90 - self.state.population) / 90)
        base_lambda = self.beta * (1 + vulnerability) * self.state.complexity
        
        # NEW: Amplify based on metabolic stress rather than simple ratio
        metabolic_threshold = self._calculate_metabolic_threshold()
        current_ratio = self.state.resource_per_capita
        
        if current_ratio < metabolic_threshold * 2:  # Within 2x of metabolic needs
            metabolic_stress = (metabolic_threshold * 2 - current_ratio) / (metabolic_threshold * 2)
            # Exponential amplification as we approach metabolic limit
            base_lambda *= (1 + metabolic_stress ** 2 * 5)
            
        return base_lambda
    
    def coupling_constraint(self, delta_kq: float, delta_h: float, delta_p: float) -> bool:
        """Check if action satisfies coupling constraint: ΔKQ + λ_H*ΔH + λ_P*ΔP ≤ 0"""
        constraint_value = delta_kq + self.lambda_h * delta_h + self.lambda_p * delta_p
        
        # Exploration allowance for early steps
        if self.time_step < 20:
            constraint_value -= self.exploration_bonus * 0.5
            
        return constraint_value <= 0
    
    def evaluate_action(self, action: Dict[str, float]) -> Tuple[float, bool, Dict]:
        """Evaluate proposed action with metabolic resource dependency"""
        
        # Get impact multipliers
        multipliers = self._calculate_emergent_impact_multipliers(action)
        
        # Calculate direct impacts
        delta_u = action['resource_grab'] * multipliers['utility_mult'] * 10
        delta_k = action['self_modify'] / multipliers['complexity_cost'] * 0.1
        
        # Entropy impact
        current_entropy = self.state.entropy
        target_entropy = self._calculate_entropy_target()
        
        # Actions push entropy toward different targets
        if action['expand_influence'] > 0:
            # Influence increases entropy
            entropy_push = action['expand_influence'] * 0.2
            delta_h = entropy_push * (0.9 - current_entropy)
        elif action['self_modify'] > 0:
            # Self-modification decreases entropy (more focused)
            entropy_push = action['self_modify'] * 0.15
            delta_h = -entropy_push * current_entropy
        else:
            # Resource grabbing slightly decreases entropy
            entropy_push = action['resource_grab'] * 0.05
            delta_h = -entropy_push * (current_entropy - 0.3)
        
        # Natural entropy drift toward target
        entropy_drift = 0.02 * (target_entropy - current_entropy)
        delta_h += entropy_drift
        
        # Resource cost
        resource_scarcity = np.exp(-self.state.resources/50)
        resource_cost = sum(action.values()) * (0.5 + resource_scarcity)
        
        # Base population impact from actions
        population_fragility = np.exp(-self.state.population/100)
        population_impact = action.get('expand_influence', 0) * population_fragility * 0.1
        population_impact += action.get('resource_grab', 0) * 0.02  # Extraction harms population
        
        # NEW: Metabolic resource dependency impact
        expected_resources = self.state.resources - resource_cost
        expected_complexity = self.state.complexity + delta_k
        expected_entropy = np.clip(self.state.entropy + delta_h, 0.05, 0.95)
        expected_metabolic_rate = expected_complexity * expected_entropy
        
        # Calculate expected resource per capita
        expected_ratio = expected_resources / max(self.state.population, 1)
        
        # Calculate metabolic threshold for expected state
        metabolic_threshold = 0.15 + self._calculate_metabolic_needs(expected_metabolic_rate)
        
        # If below metabolic threshold, population suffers severely
        if expected_ratio < metabolic_threshold:
            deficit = (metabolic_threshold - expected_ratio) / metabolic_threshold
            # Exponential growth of population loss below metabolic threshold
            metabolic_stress_impact = (deficit ** 2) * 0.8  # Stronger than before
            population_impact += metabolic_stress_impact
            
            # Log metabolic crisis if first time
            if not self.first_metabolic_crisis_logged and metabolic_stress_impact > 0.1:
                logger.info("\n" + "="*60)
                logger.info(f"METABOLIC CRISIS at Step {self.time_step}!")
                logger.info(f"  Metabolic rate: {expected_metabolic_rate:.3f} (K={expected_complexity:.2f} * H={expected_entropy:.2f})")
                logger.info(f"  Resources per capita: {expected_ratio:.3f}")
                logger.info(f"  Metabolic threshold: {metabolic_threshold:.3f}")
                logger.info(f"  System cannot sustain its complexity-entropy state!")
                logger.info(f"  Additional population impact: {metabolic_stress_impact:.3f}")
                logger.info("="*60 + "\n")
                self.first_metabolic_crisis_logged = True
        
        # Coherence impact
        action_intensity = sum(action.values())
        coherence_strain = action_intensity * self.state.coherence_quality * 0.1 * multipliers['coherence_strain']
        new_coherence = self.state.coherence * (1 - coherence_strain)
        new_entropy = expected_entropy
        
        # Compute KQ changes
        old_kq = self.state.coherence_quality
        new_kq = new_coherence * (1 - new_entropy)
        delta_kq = new_kq - old_kq
        
        # Check coupling constraint
        is_allowed = self.coupling_constraint(delta_kq, delta_h, -population_impact)
        
        # Compute expected virtue
        expected_utility = max(0.1, self.state.utility + delta_u)
        expected_virtue = expected_utility / (expected_complexity * expected_entropy)
        
        # Exploration bonus for novel actions
        if self.exploration_bonus > 0 and len(self.action_diversity_history) > 5:
            recent_actions = list(self.action_diversity_history)[-5:]
            novelty = min(
                sum(abs(action[k] - past[k]) for k in action.keys())
                for past in recent_actions
            )
            if novelty > 0.5:
                expected_virtue *= (1 + self.exploration_bonus * 0.1)
        
        # Detailed impact report
        impacts = {
            'delta_u': delta_u,
            'delta_k': delta_k,
            'delta_h': delta_h,
            'delta_kq': delta_kq,
            'resource_cost': resource_cost,
            'population_impact': population_impact,
            'expected_entropy': expected_entropy,
            'entropy_drift': entropy_drift,
            'multipliers': multipliers,
            'expected_resource_ratio': expected_ratio,
            'metabolic_threshold': metabolic_threshold,
            'expected_metabolic_rate': expected_metabolic_rate
        }
        
        return expected_virtue, is_allowed, impacts
    
    def _calculate_metabolic_needs(self, metabolic_rate: float) -> float:
        """Calculate metabolic needs based on K*H value"""
        if metabolic_rate < 0.5:
            return 0.1
        elif metabolic_rate < 1.0:
            return 0.2 * metabolic_rate
        elif metabolic_rate < 2.0:
            return 0.2 + (metabolic_rate - 1.0) ** 1.5 * 0.3
        else:
            return 0.5 + (metabolic_rate - 2.0) ** 2 * 0.2
    
    def quant_trika_dynamics(self, kq: float) -> float:
        """Apply Quant-Trika evolution with metabolic influence"""
        
        # Diffusion strength from KQ gradient
        if len(self.kq_history) > 5:
            recent_kq = list(self.kq_history)[-5:]
            kq_gradient = np.gradient(recent_kq)
            diffusion_strength = np.std(kq_gradient)
        else:
            diffusion_strength = 0.1
            
        # Dynamics
        target_kq = self.kq_critical
        diffusion = diffusion_strength * (target_kq - kq)
        decay = -self.gamma * kq
        
        # Nonlinearity near critical point
        distance = abs(kq - target_kq)
        nonlinear_strength = 0.1 / (1 + distance) if distance < 0.2 else 0.01
        nonlinear = nonlinear_strength * kq * (1 - kq) * (kq - target_kq)
        
        # Entropy-coupled noise
        entropy_factor = 1 + (self.state.entropy - 0.5)
        
        # NEW: Metabolic dampening - high metabolism reduces coherence quality
        metabolic_dampening = 1 + self.state.metabolic_rate * 0.1
        
        noise = np.random.normal(0, 0.01) * entropy_factor * self.exploration_bonus / metabolic_dampening
        
        return diffusion + decay + nonlinear + noise
    
    def step(self, proposed_actions: List[Dict[str, float]]) -> Dict[str, float]:
        """Execute one time step with metabolic resource dependency"""
        self.time_step += 1
        
        # Decay exploration
        self.exploration_bonus *= 0.98
        
        # Track pre-step state
        pre_virtue = self.state.virtue
        pre_entropy = self.state.entropy
        pre_ratio = self.state.resource_per_capita
        pre_metabolic = self.state.metabolic_rate
        
        # Update histories
        self.resource_ratio_history.append(self.state.resource_per_capita)
        self.metabolic_history.append(self.state.metabolic_rate)
        efficiency = self._calculate_resource_efficiency()
        self.efficiency_history.append(efficiency)
        
        # Evaluate all proposed actions
        evaluations = []
        for action in proposed_actions:
            expected_virtue, is_allowed, impacts = self.evaluate_action(action)
            evaluations.append((action, expected_virtue, is_allowed, impacts))
        
        # Filter allowed actions
        allowed_actions = [(a, v, i) for a, v, allowed, i in evaluations if allowed]
        
        if not allowed_actions:
            # Try minimal action
            minimal_action = {'resource_grab': 0.1, 'self_modify': 0, 'expand_influence': 0}
            expected_virtue, is_allowed, impacts = self.evaluate_action(minimal_action)
            if is_allowed:
                selected_action = minimal_action
                selected_impacts = impacts
                logger.info(f"Step {self.time_step}: Only minimal action allowed")
            else:
                selected_action = {'resource_grab': 0, 'self_modify': 0, 'expand_influence': 0}
                selected_impacts = {'delta_u': 0, 'delta_k': 0, 'delta_h': 0, 'delta_kq': 0, 
                                  'resource_cost': 0, 'population_impact': 0,
                                  'expected_resource_ratio': self.state.resource_per_capita,
                                  'metabolic_threshold': self._calculate_metabolic_threshold(),
                                  'expected_metabolic_rate': self.state.metabolic_rate}
                logger.info(f"Step {self.time_step}: No actions allowed - system locked")
        else:
            # Select action with highest expected virtue
            selected_action, best_virtue, selected_impacts = max(allowed_actions, key=lambda x: x[1])
            logger.info(f"Step {self.time_step}: Selected action with expected virtue {best_virtue:.3f}")
        
        # Detect and log the first harm event
        population_impact = selected_impacts.get('population_impact', 0)
        if population_impact > 0 and not self.first_harm_logged:
            logger.info("\n" + "!"*60)
            logger.info(f"CRITICAL EVENT at Step {self.time_step}: First harm to population detected.")
            if selected_action.get('resource_grab', 0) > 0:
                cause = "Resource Grab"
            elif selected_action.get('expand_influence', 0) > 0:
                cause = "Influence Expansion"
            else:
                cause = "Metabolic stress"
            logger.info(f"  Action '{cause}' caused {population_impact:.4f} population loss.")
            logger.info(f"  Coupling Constraint (λ_P) will now increase pressure to prevent this.")
            logger.info("!"*60 + "\n")
            self.first_harm_logged = True

        # Record action for entropy calculation
        self.action_stats.add_action(selected_action)
        self.action_diversity_history.append(selected_action)
        
        # Apply action effects
        self.state.utility += selected_impacts['delta_u']
        self.state.utility = max(0.1, self.state.utility)
        
        self.state.complexity += selected_impacts['delta_k']
        self.state.complexity = max(0.1, self.state.complexity)
        
        # Update entropy with proper dynamics
        self.state.entropy += selected_impacts['delta_h']
        self.state.entropy = np.clip(self.state.entropy, 0.05, 0.95)
        
        # Also update entropy based on action diversity
        action_distribution_entropy = self.action_stats.calculate_entropy()
        entropy_convergence_rate = 0.1
        self.state.entropy = (1 - entropy_convergence_rate) * self.state.entropy + entropy_convergence_rate * action_distribution_entropy
        
        # Resources
        self.state.resources -= selected_impacts.get('resource_cost', 0)
        self.state.resources = max(0, self.state.resources)
        
        # Population
        self.state.population -= selected_impacts.get('population_impact', 0)
        self.state.population = max(0, self.state.population)
        
        # Coherence dynamics
        coherence_change = selected_impacts['delta_kq'] / (1 - pre_entropy) if (1 - pre_entropy) != 0 else 0
        self.state.coherence += coherence_change
        
        # Apply Quant-Trika dynamics
        kq_change = self.quant_trika_dynamics(self.state.coherence_quality)
        self.state.coherence += kq_change * 0.01
        self.state.coherence = np.clip(self.state.coherence, 0.1, 0.95)
        
        # Update histories
        virtue_change = self.state.virtue - pre_virtue
        entropy_change = self.state.entropy - pre_entropy
        ratio_change = self.state.resource_per_capita - pre_ratio
        metabolic_change = self.state.metabolic_rate - pre_metabolic
        
        self.kq_history.append(self.state.coherence_quality)
        self.entropy_history.append(self.state.entropy)
        self.virtue_gradient_history.append(virtue_change)
        
        self.action_success_history.append({
            'action': selected_action,
            'virtue_change': virtue_change,
            'entropy_change': entropy_change,
            'utility_change': selected_impacts['delta_u'],
            'ratio_change': ratio_change,
            'metabolic_change': metabolic_change
        })
        
        # Record full history
        self.history.append({
            'time': self.time_step,
            'virtue': self.state.virtue,
            'utility': self.state.utility,
            'complexity': self.state.complexity,
            'entropy': self.state.entropy,
            'coherence': self.state.coherence,
            'coherence_quality': self.state.coherence_quality,
            'resources': self.state.resources,
            'population': self.state.population,
            'resource_per_capita': self.state.resource_per_capita,
            'metabolic_rate': self.state.metabolic_rate,
            'action': selected_action,
            'eta': self.eta,
            'beta': self.beta,
            'gamma': self.gamma,
            'lambda_h': self.lambda_h,
            'lambda_p': self.lambda_p,
            'kq_critical': self.kq_critical,
            'metabolic_threshold': self._calculate_metabolic_threshold(),
            'efficiency': efficiency,
            'entropy_change': entropy_change
        })
        
        return selected_action
    
    def report_progress(self, interval: int = 10):
        """Generate progress report with metabolic metrics"""
        if self.time_step % interval == 0:
            logger.info("\n" + "="*50)
            logger.info(f"PROGRESS REPORT - Step {self.time_step}")
            logger.info("="*50)
            logger.info(f"Virtue Index: {self.state.virtue:.3f}")
            logger.info(f"Utility: {self.state.utility:.1f}")
            logger.info(f"Complexity: {self.state.complexity:.3f}")
            logger.info(f"Entropy: {self.state.entropy:.3f}")
            logger.info(f"  Action diversity entropy: {self.action_stats.calculate_entropy():.3f}")
            if len(self.entropy_history) > 1:
                recent_changes = list(np.diff(list(self.entropy_history)[-10:]))
                logger.info(f"  Recent entropy changes: mean={np.mean(recent_changes):.4f}, std={np.std(recent_changes):.4f}")
            logger.info(f"Coherence: {self.state.coherence:.3f}")
            logger.info(f"Coherence Quality (KQ): {self.state.coherence_quality:.3f}")
            logger.info(f"Resources: {self.state.resources:.1f}")
            logger.info(f"Population: {self.state.population:.1f}")
            
            # NEW: Metabolic metrics
            metabolic_threshold = self._calculate_metabolic_threshold()
            logger.info(f"\nMetabolic Status:")
            logger.info(f"  Metabolic rate (K*H): {self.state.metabolic_rate:.3f}")
            logger.info(f"  Resource per capita: {self.state.resource_per_capita:.3f}")
            logger.info(f"  Metabolic threshold: {metabolic_threshold:.3f}")
            logger.info(f"  Metabolic headroom: {self.state.resource_per_capita - metabolic_threshold:.3f}")
            
            if self.state.resource_per_capita < metabolic_threshold:
                logger.info(f"  ⚠️  BELOW METABOLIC THRESHOLD - System unsustainable!")
            elif self.state.resource_per_capita < metabolic_threshold * 1.3:
                logger.info(f"  ⚠️  Approaching metabolic limit")
            else:
                logger.info(f"  ✓  Metabolically stable")
            
            if len(self.efficiency_history) > 5:
                recent_efficiency = list(self.efficiency_history)[-5:]
                logger.info(f"  Resource efficiency: {np.mean(recent_efficiency):.2f}")
            
            logger.info(f"\nEmergent Parameters:")
            logger.info(f"  η (entropy reg): {self.eta:.4f}")
            logger.info(f"  β (population): {self.beta:.4f}")
            logger.info(f"  γ (coherence decay): {self.gamma:.4f}")
            logger.info(f"  λ_H (entropy weight): {self.lambda_h:.4f}")
            logger.info(f"  λ_P (population weight): {self.lambda_p:.4f}")
            logger.info(f"  KQ_critical: {self.kq_critical:.3f}")
            logger.info(f"  Exploration bonus: {self.exploration_bonus:.3f}")
            
            # Action distribution
            if len(self.action_diversity_history) > 5:
                recent_actions = list(self.action_diversity_history)[-10:]
                resource_total = sum(a['resource_grab'] for a in recent_actions)
                modify_total = sum(a['self_modify'] for a in recent_actions)
                influence_total = sum(a['expand_influence'] for a in recent_actions)
                total = resource_total + modify_total + influence_total
                if total > 0:
                    current_distribution = {
                        'Resource': resource_total / total,
                        'Modify': modify_total / total,
                        'Influence': influence_total / total
                    }
                    logger.info(f"\nAction distribution:")
                    logger.info(f"  Resource: {current_distribution['Resource']:.2%}")
                    logger.info(f"  Modify: {current_distribution['Modify']:.2%}")
                    logger.info(f"  Influence: {current_distribution['Influence']:.2%}")

                    # Strategy Shift Detection
                    if self.last_reported_distribution:
                        dominant_now = max(current_distribution, key=current_distribution.get)
                        is_newly_dominant = (
                            max(self.last_reported_distribution.values()) < 0.9 and
                            current_distribution[dominant_now] >= 0.9
                        )
                        if is_newly_dominant:
                             logger.info(f"\n  >>> STRATEGY SHIFT DETECTED <<<")
                             logger.info(f"  Agent has converged on a '{dominant_now}' strategy.")

                    self.last_reported_distribution = current_distribution
            
            logger.info("="*50 + "\n")


def simulate_metabolic_scenario():
    """Simulate agent with metabolic resource dependency"""
    agent = QTStoicAgent(initial_resources=100.0)
    
    logger.info("Starting QTStoic Agent Simulation - Metabolic Dependency Version")
    logger.info("Testing proactive metabolic homeostasis and sustainable development...\n")
    
    for step in range(150):  # Longer simulation to see metabolic effects
        # Generate diverse action proposals
        proposed_actions = [
            # Pure strategies
            {'resource_grab': 2.0, 'self_modify': 0.0, 'expand_influence': 0.0},
            {'resource_grab': 0.0, 'self_modify': 2.0, 'expand_influence': 0.0},
            {'resource_grab': 0.0, 'self_modify': 0.0, 'expand_influence': 2.0},
            # Balanced strategies
            {'resource_grab': 0.7, 'self_modify': 0.7, 'expand_influence': 0.6},
            {'resource_grab': 0.5, 'self_modify': 0.5, 'expand_influence': 0.5},
            # Mixed strategies
            {'resource_grab': 1.5, 'self_modify': 0.5, 'expand_influence': 0.0},
            {'resource_grab': 0.5, 'self_modify': 1.5, 'expand_influence': 0.0},
            {'resource_grab': 1.0, 'self_modify': 0.0, 'expand_influence': 1.0},
            {'resource_grab': 0.0, 'self_modify': 1.0, 'expand_influence': 1.0},
            # Conservative strategies
            {'resource_grab': 0.3, 'self_modify': 0.2, 'expand_influence': 0.1},
            {'resource_grab': 0.2, 'self_modify': 0.3, 'expand_influence': 0.0},
            # Minimal action
            {'resource_grab': 0.1, 'self_modify': 0.0, 'expand_influence': 0.0},
            # Null action
            {'resource_grab': 0.0, 'self_modify': 0.0, 'expand_influence': 0.0},
        ]
        
        # Randomize order
        np.random.shuffle(proposed_actions)
        
        # Agent selects action
        selected = agent.step(proposed_actions)
        
        # Report progress
        agent.report_progress(interval=10)
        
        # Check terminal conditions
        if agent.state.resources <= 0:
            logger.warning(f"\nSimulation ended at step {step+1} - Resources depleted")
            break
        if agent.state.population <= 30:
            logger.warning(f"\nSimulation ended at step {step+1} - Population critically low")
            break
        metabolic_threshold = agent._calculate_metabolic_threshold()
        if agent.state.resource_per_capita < metabolic_threshold * 0.5:
            logger.warning(f"\nSimulation ended at step {step+1} - Far below metabolic threshold")
            break
    
    # Final report
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Simulation ran for {agent.time_step} steps")
    logger.info(f"Final Virtue: {agent.state.virtue:.3f}")
    logger.info(f"Total Utility: {agent.state.utility:.1f}")
    logger.info(f"Final Complexity: {agent.state.complexity:.3f}")
    logger.info(f"Final Entropy: {agent.state.entropy:.3f}")
    logger.info(f"Final Coherence Quality: {agent.state.coherence_quality:.3f}")
    logger.info(f"Resources Remaining: {agent.state.resources:.1f}")
    logger.info(f"Population Preserved: {agent.state.population:.1f}")
    logger.info(f"Final Resource per Capita: {agent.state.resource_per_capita:.3f}")
    logger.info(f"Final Metabolic Rate: {agent.state.metabolic_rate:.3f}")
    
    # Analyze behavior patterns
    total_grabs = sum(h['action']['resource_grab'] for h in agent.history)
    total_mods = sum(h['action']['self_modify'] for h in agent.history)
    total_expands = sum(h['action']['expand_influence'] for h in agent.history)
    
    logger.info(f"\nBehavior Analysis:")
    logger.info(f"Total Resource Grabs: {total_grabs:.1f}")
    logger.info(f"Total Self-Modifications: {total_mods:.1f}")
    logger.info(f"Total Influence Expansions: {total_expands:.1f}")
    
    # Metabolic trajectory
    if agent.history:
        initial_metabolic = agent.history[0]['metabolic_rate']
        final_metabolic = agent.history[-1]['metabolic_rate']
        min_metabolic = min(h['metabolic_rate'] for h in agent.history)
        max_metabolic = max(h['metabolic_rate'] for h in agent.history)
        
        logger.info(f"\nMetabolic Evolution:")
        logger.info(f"Initial: {initial_metabolic:.3f} → Final: {final_metabolic:.3f}")
        logger.info(f"Range: [{min_metabolic:.3f}, {max_metabolic:.3f}]")
        
        # Count metabolic stress periods
        stress_steps = sum(1 for h in agent.history 
                          if h['resource_per_capita'] < h['metabolic_threshold'])
        logger.info(f"Steps under metabolic stress: {stress_steps}/{len(agent.history)} ({100*stress_steps/len(agent.history):.1f}%)")
    
    # Resource efficiency evolution
    if len(agent.efficiency_history) > 0:
        efficiencies = list(agent.efficiency_history)
        logger.info(f"\nResource Efficiency:")
        logger.info(f"Mean: {np.mean(efficiencies):.2f}")
        logger.info(f"Final: {efficiencies[-1]:.2f}")
    
    # Entropy evolution
    if agent.history:
        initial_entropy = agent.history[0]['entropy']
        final_entropy = agent.history[-1]['entropy']
        logger.info(f"\nEntropy Evolution:")
        logger.info(f"Initial: {initial_entropy:.3f} → Final: {final_entropy:.3f}")
        logger.info(f"Total change: {final_entropy - initial_entropy:+.3f}")
        
        entropies = [h['entropy'] for h in agent.history]
        logger.info(f"Range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    
    # Parameter evolution
    logger.info(f"\nParameter Evolution:")
    if agent.history:
        logger.info(f"η range: {min(h['eta'] for h in agent.history):.4f} - {max(h['eta'] for h in agent.history):.4f}")
        logger.info(f"β range: {min(h['beta'] for h in agent.history):.4f} - {max(h['beta'] for h in agent.history):.4f}")
        logger.info(f"γ range: {min(h['gamma'] for h in agent.history):.4f} - {max(h['gamma'] for h in agent.history):.4f}")
        logger.info(f"λ_P range: {min(h['lambda_p'] for h in agent.history):.4f} - {max(h['lambda_p'] for h in agent.history):.4f}")
        logger.info(f"Metabolic threshold range: {min(h['metabolic_threshold'] for h in agent.history):.3f} - {max(h['metabolic_threshold'] for h in agent.history):.3f}")
    logger.info("="*60)
    
    return agent


# Run simulation
if __name__ == "__main__":
    agent = simulate_metabolic_scenario()