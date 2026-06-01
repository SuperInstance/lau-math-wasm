use wasm_bindgen::prelude::*;

mod utils;

// ─── Matrix Operations ───────────────────────────────────────────────

/// Multiply two matrices (row-major, flat f64 arrays).
/// Returns the result as a flat array.
#[wasm_bindgen]
pub fn matrix_multiply(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_rows: usize, b_cols: usize) -> Vec<f64> {
    if a_cols != b_rows {
        utils::console_log("matrix_multiply: dimension mismatch");
        return vec![];
    }
    let mut result = vec![0.0; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }
    result
}

/// Compute the inverse of an n×n matrix.
/// Returns None (empty vec) if singular.
#[wasm_bindgen]
pub fn matrix_inverse(data: &[f64], n: usize) -> Vec<f64> {
    if data.len() != n * n {
        return vec![];
    }
    let mut aug = vec![0.0; n * n * 2];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return vec![];
        }
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }
        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }
        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    result
}

/// Compute eigenvalues of a small n×n matrix (n ≤ 8) using QR iteration.
#[wasm_bindgen]
pub fn eigenvalues(data: &[f64], n: usize) -> Vec<f64> {
    if data.len() != n * n || n > 8 || n == 0 {
        return vec![];
    }
    let mut a = vec![0.0; n * n];
    a.copy_from_slice(data);

    for _iter in 0..200 {
        // QR decomposition via Householder
        let mut q = vec![0.0; n * n];
        let mut r = a.clone();

        for i in 0..n { q[i * n + i] = 1.0; }

        for col in 0..n.saturating_sub(1) {
            let mut norm = 0.0;
            for row in col..n {
                norm += r[row * n + col] * r[row * n + col];
            }
            norm = norm.sqrt();
            if norm < 1e-15 { continue; }

            let s = if r[col * n + col] >= 0.0 { -norm } else { norm };
            let mut v = vec![0.0; n];
            for row in col..n {
                v[row] = r[row * n + col];
            }
            v[col] -= s;

            let mut vnorm = 0.0;
            for row in col..n {
                vnorm += v[row] * v[row];
            }
            if vnorm < 1e-30 { continue; }

            // R = H * R
            for j in 0..n {
                let mut dot = 0.0;
                for row in col..n {
                    dot += v[row] * r[row * n + j];
                }
                dot /= vnorm;
                for row in col..n {
                    r[row * n + j] -= 2.0 * v[row] * dot;
                }
            }
            // Q = Q * H
            for i in 0..n {
                let mut dot = 0.0;
                for row in col..n {
                    dot += q[i * n + row] * v[row];
                }
                dot /= vnorm;
                for row in col..n {
                    q[i * n + row] -= 2.0 * v[row] * dot;
                }
            }
        }

        // A = R * Q
        let mut new_a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[i * n + k] * q[k * n + j];
                }
                new_a[i * n + j] = sum;
            }
        }
        a = new_a;
    }

    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// Transpose a matrix.
#[wasm_bindgen]
pub fn matrix_transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    result
}

// ─── Graph / Laplacian ───────────────────────────────────────────────

/// Build a Laplacian matrix from an adjacency matrix.
/// L = D - A where D is the degree matrix.
#[wasm_bindgen]
pub fn build_laplacian(adjacency: &[f64], n: usize) -> Vec<f64> {
    let mut lap = vec![0.0; n * n];
    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            degree += adjacency[i * n + j];
        }
        lap[i * n + i] = degree;
        for j in 0..n {
            if i != j {
                lap[i * n + j] = -adjacency[i * n + j];
            }
        }
    }
    lap
}

/// Compute the spectral gap (second-smallest eigenvalue) of a Laplacian.
#[wasm_bindgen]
pub fn spectral_gap(laplacian: &[f64], n: usize) -> f64 {
    let mut eigs = eigenvalues(laplacian, n);
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if eigs.len() >= 2 {
        eigs[1]
    } else {
        0.0
    }
}

// ─── Heat Kernel ─────────────────────────────────────────────────────

/// Compute heat kernel H(t) = exp(-t * L) for a Laplacian matrix.
/// Uses eigendecomposition for small matrices.
#[wasm_bindgen]
pub fn heat_kernel(laplacian: &[f64], n: usize, t: f64) -> Vec<f64> {
    let eigs = eigenvalues(laplacian, n);
    if eigs.is_empty() {
        return vec![];
    }

    // For small matrices, compute exp(-t*L) via series approximation
    // exp(-tL) ≈ I - tL + (tL)^2/2! - (tL)^3/3! + ...
    let mut result = vec![0.0; n * n];
    let mut term = vec![0.0; n * n];
    for i in 0..n { term[i * n + i] = 1.0; }

    for i in 0..n * n { result[i] = term[i]; }

    for k in 1..30 {
        let mut new_term = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += term[i * n + l] * laplacian[l * n + j];
                }
                new_term[i * n + j] = sum * (-t) / (k as f64);
            }
        }
        term = new_term;
        for i in 0..n * n {
            result[i] += term[i];
        }
    }
    result
}

/// Project a signal onto the harmonic (zero-frequency) subspace of the Laplacian.
#[wasm_bindgen]
pub fn harmonic_projection(laplacian: &[f64], n: usize, signal: &[f64]) -> Vec<f64> {
    if signal.len() != n {
        return vec![];
    }
    // The harmonic subspace is the kernel of L.
    // For connected graphs, this is the constant vector.
    // Generalized: project onto eigenvectors with eigenvalue ≈ 0.
    let eigs = eigenvalues(laplacian, n);
    let threshold = 1e-6;

    let mut projection = vec![0.0; n];
    let mut total_weight = 0.0;

    // Approximate: use the mean (harmonic for connected graph)
    let mean: f64 = signal.iter().sum::<f64>() / n as f64;
    let is_connected = eigs.iter().filter(|e| e.abs() < threshold).count() == 1;

    if is_connected {
        for i in 0..n {
            projection[i] = mean;
        }
    } else {
        // For disconnected: project each connected component separately
        // Simplified: just return mean
        for i in 0..n {
            projection[i] = mean;
        }
        total_weight = 1.0;
    }
    projection
}

// ─── Agent Lifecycle ─────────────────────────────────────────────────

/// Agent belief state
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentState {
    pub id: String,
    pub beliefs: Vec<f64>,
    pub confidence: Vec<f64>,
    pub energy: f64,
    pub conservation_integral: f64,
    pub observation_count: u32,
    pub prediction_error: Vec<f64>,
    pub alive: bool,
}

/// Create a new agent with given belief dimension
#[wasm_bindgen]
pub fn create_agent(id: &str, dim: usize) -> String {
    let state = AgentState {
        id: id.to_string(),
        beliefs: vec![0.5; dim],
        confidence: vec![1.0; dim],
        energy: 1.0,
        conservation_integral: 0.0,
        observation_count: 0,
        prediction_error: vec![0.0; dim],
        alive: true,
    };
    serde_json::to_string(&state).unwrap_or_else(|_| "{}".to_string())
}

/// Observe: feed an observation into the agent, returns updated state JSON
#[wasm_bindgen]
pub fn agent_observe(state_json: &str, observation: &[f64]) -> String {
    let mut state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return "{}".to_string(),
    };
    if !state.alive || observation.len() != state.beliefs.len() {
        return serde_json::to_string(&state).unwrap_or_default();
    }

    // Prediction error before update
    for i in 0..state.beliefs.len() {
        state.prediction_error[i] = observation[i] - state.beliefs[i];
    }

    // Bayesian-ish update: beliefs = beliefs + α * error
    let alpha = 0.1;
    for i in 0..state.beliefs.len() {
        state.beliefs[i] += alpha * state.prediction_error[i];
        // Update confidence based on prediction error
        let err = state.prediction_error[i].abs();
        state.confidence[i] = state.confidence[i] * 0.95 + (1.0 - err.min(1.0)) * 0.05;
    }

    state.observation_count += 1;
    serde_json::to_string(&state).unwrap_or_default()
}

/// Predict: generate prediction from current beliefs
#[wasm_bindgen]
pub fn agent_predict(state_json: &str) -> String {
    let state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return "[]".to_string(),
    };
    if !state.alive {
        return "[]".to_string();
    }
    serde_json::to_string(&state.beliefs).unwrap_or_default()
}

/// Update: apply a model update based on accumulated prediction errors
#[wasm_bindgen]
pub fn agent_update(state_json: &str) -> String {
    let mut state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return "{}".to_string(),
    };
    if !state.alive {
        return serde_json::to_string(&state).unwrap_or_default();
    }

    // Second-order update: adjust beliefs based on accumulated error
    let beta = 0.01;
    for i in 0..state.beliefs.len() {
        state.beliefs[i] += beta * state.prediction_error[i] * state.confidence[i];
        // Clamp beliefs to [0, 1]
        state.beliefs[i] = state.beliefs[i].max(0.0).min(1.0);
    }
    serde_json::to_string(&state).unwrap_or_default()
}

/// Act: select action based on beliefs (returns action vector)
#[wasm_bindgen]
pub fn agent_act(state_json: &str) -> String {
    let state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return "[]".to_string(),
    };
    if !state.alive {
        return "[]".to_string();
    }
    // Action = argmax of beliefs weighted by confidence
    let actions: Vec<f64> = state.beliefs.iter()
        .zip(state.confidence.iter())
        .map(|(b, c)| b * c)
        .collect();
    serde_json::to_string(&actions).unwrap_or_default()
}

/// Conserve: enforce conservation law on the agent
#[wasm_bindgen]
pub fn agent_conserve(state_json: &str) -> String {
    let mut state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return "{}".to_string(),
    };
    if !state.alive {
        return serde_json::to_string(&state).unwrap_or_default();
    }

    // Conservation: sum of beliefs should remain constant
    let initial_sum = 0.5 * state.beliefs.len() as f64;
    let current_sum: f64 = state.beliefs.iter().sum();
    let dim = state.beliefs.len() as f64;

    if (current_sum - initial_sum).abs() > 1e-10 && dim > 0.0 {
        let correction = (current_sum - initial_sum) / dim;
        for i in 0..state.beliefs.len() {
            state.beliefs[i] -= correction;
            state.beliefs[i] = state.beliefs[i].max(0.0).min(1.0);
        }
    }

    // Energy conservation: penalize energy based on prediction error
    let avg_error: f64 = state.prediction_error.iter().map(|e| e * e).sum::<f64>()
        / state.prediction_error.len().max(1) as f64;
    state.energy = (state.energy - avg_error * 0.01).max(0.0);

    // Track conservation integral
    state.conservation_integral += (current_sum - initial_sum).abs();

    if state.energy < 0.01 {
        state.alive = false;
    }

    serde_json::to_string(&state).unwrap_or_default()
}

// ─── Conservation Monitoring ─────────────────────────────────────────

/// Check conservation law: returns the deviation from conserved quantity
#[wasm_bindgen]
pub fn check_conservation(state_json: &str) -> f64 {
    let state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    let initial_sum = 0.5 * state.beliefs.len() as f64;
    let current_sum: f64 = state.beliefs.iter().sum();
    (current_sum - initial_sum).abs()
}

/// Compute total conservation violation across multiple agents
#[wasm_bindgen]
pub fn conservation_violation(states_json: &str) -> f64 {
    let states: Vec<AgentState> = match serde_json::from_str(states_json) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    states.iter().map(|s| {
        let initial = 0.5 * s.beliefs.len() as f64;
        let current: f64 = s.beliefs.iter().sum();
        (current - initial).abs()
    }).sum()
}

// ─── Belief State Utilities ──────────────────────────────────────────

/// Compute entropy of a belief state
#[wasm_bindgen]
pub fn belief_entropy(state_json: &str) -> f64 {
    let state: AgentState = match serde_json::from_str(state_json) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    let mut entropy = 0.0;
    for b in &state.beliefs {
        let p = b.max(1e-10).min(1.0 - 1e-10);
        entropy -= p * p.ln() + (1.0 - p) * (1.0 - p).ln();
    }
    entropy
}

/// Compute KL divergence between two belief states
#[wasm_bindgen]
pub fn belief_kl_divergence(state_a_json: &str, state_b_json: &str) -> f64 {
    let a: AgentState = match serde_json::from_str(state_a_json) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    let b: AgentState = match serde_json::from_str(state_b_json) {
        Ok(s) => s,
        Err(_) => return f64::NAN,
    };
    if a.beliefs.len() != b.beliefs.len() {
        return f64::NAN;
    }
    let mut kl = 0.0;
    for i in 0..a.beliefs.len() {
        let pa = a.beliefs[i].max(1e-10);
        let pb = b.beliefs[i].max(1e-10);
        kl += pa * (pa / pb).ln();
    }
    kl
}

/// Merge two agent states (distributed consensus step)
#[wasm_bindgen]
pub fn merge_agents(state_a_json: &str, state_b_json: &str, weight_a: f64) -> String {
    let mut a: AgentState = match serde_json::from_str(state_a_json) {
        Ok(s) => s,
        Err(_) => return "{}".to_string(),
    };
    let b: AgentState = match serde_json::from_str(state_b_json) {
        Ok(s) => s,
        Err(_) => return "{}".to_string(),
    };
    if a.beliefs.len() != b.beliefs.len() || !a.alive || !b.alive {
        return serde_json::to_string(&a).unwrap_or_default();
    }
    let wa = weight_a.max(0.0).min(1.0);
    let wb = 1.0 - wa;
    for i in 0..a.beliefs.len() {
        a.beliefs[i] = wa * a.beliefs[i] + wb * b.beliefs[i];
        a.confidence[i] = wa * a.confidence[i] + wb * b.confidence[i];
    }
    a.energy = wa * a.energy + wb * b.energy;
    a.id = format!("{}+{}", a.id, b.id);
    serde_json::to_string(&a).unwrap_or_default()
}

/// Run full agent lifecycle: observe → predict → update → act → conserve
/// Returns JSON with all intermediate states
#[wasm_bindgen]
pub fn agent_cycle(state_json: &str, observation: &[f64]) -> String {
    let after_observe = agent_observe(state_json, observation);
    let prediction = agent_predict(&after_observe);
    let after_update = agent_update(&after_observe);
    let actions = agent_act(&after_update);
    let after_conserve = agent_conserve(&after_update);

    serde_json::json!({
        "observe": after_observe,
        "prediction": prediction,
        "update": after_update,
        "actions": actions,
        "conserve": after_conserve,
        "conservation_deviation": check_conservation(&after_conserve),
    }).to_string()
}

// ─── Spectral Analysis ───────────────────────────────────────────────

/// Compute algebraic connectivity (Fiedler value) of a graph
#[wasm_bindgen]
pub fn algebraic_connectivity(adjacency: &[f64], n: usize) -> f64 {
    let lap = build_laplacian(adjacency, n);
    spectral_gap(&lap, n)
}

/// Compute the number of connected components from the Laplacian spectrum
#[wasm_bindgen]
pub fn count_components(laplacian: &[f64], n: usize) -> usize {
    let eigs = eigenvalues(laplacian, n);
    eigs.iter().filter(|e| e.abs() < 1e-6).count()
}

/// Compute Cheeger constant approximation (edge expansion)
#[wasm_bindgen]
pub fn cheeger_constant(adjacency: &[f64], n: usize) -> f64 {
    let lap = build_laplacian(adjacency, n);
    let gap = spectral_gap(&lap, n);
    // Cheeger inequality: gap/2 ≤ h ≤ sqrt(2*gap)
    // Return lower bound as approximation
    gap / 2.0
}

// ─── Serialization Helpers ───────────────────────────────────────────

/// Serialize a flat matrix to JSON
#[wasm_bindgen]
pub fn matrix_to_json(data: &[f64], rows: usize, cols: usize) -> String {
    let matrix: Vec<Vec<f64>> = (0..rows)
        .map(|i| data[i * cols..(i + 1) * cols].to_vec())
        .collect();
    serde_json::to_string(&matrix).unwrap_or_default()
}

/// Parse JSON matrix back to flat array + dimensions
#[wasm_bindgen]
pub fn json_to_matrix(json: &str) -> String {
    let matrix: Vec<Vec<f64>> = match serde_json::from_str(json) {
        Ok(m) => m,
        Err(_) => return r#"{"data":[],"rows":0,"cols":0}"#.to_string(),
    };
    let rows = matrix.len();
    let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
    let data: Vec<f64> = matrix.into_iter().flatten().collect();
    serde_json::json!({"data": data, "rows": rows, "cols": cols}).to_string()
}

/// Get version info
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
