use wasm_bindgen_test::*;

use lau_math_wasm::*;

wasm_bindgen_test_configure!(run_in_browser);

// ─── Matrix Tests ────────────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_matrix_multiply_identity() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    let result = matrix_multiply(&a, 2, 2, &eye, 2, 2);
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 2.0).abs() < 1e-10);
    assert!((result[2] - 3.0).abs() < 1e-10);
    assert!((result[3] - 4.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_matrix_multiply_2x3_3x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let result = matrix_multiply(&a, 2, 3, &b, 3, 2);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 58.0).abs() < 1e-10);
    assert!((result[1] - 64.0).abs() < 1e-10);
    assert!((result[2] - 139.0).abs() < 1e-10);
    assert!((result[3] - 154.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_matrix_inverse_identity() {
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    let inv = matrix_inverse(&eye, 2);
    assert_eq!(inv.len(), 4);
    assert!((inv[0] - 1.0).abs() < 1e-10);
    assert!((inv[3] - 1.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_matrix_inverse_3x3() {
    let m = vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let inv = matrix_inverse(&m, 3);
    assert_eq!(inv.len(), 9);
    // Verify M * M^-1 ≈ I
    let result = matrix_multiply(&m, 3, 3, &inv, 3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((result[i * 3 + j] - expected).abs() < 1e-8, "Mismatch at ({},{})", i, j);
        }
    }
}

#[wasm_bindgen_test]
fn test_matrix_inverse_singular() {
    let singular = vec![1.0, 2.0, 2.0, 4.0];
    let inv = matrix_inverse(&singular, 2);
    assert!(inv.is_empty());
}

#[wasm_bindgen_test]
fn test_matrix_transpose() {
    let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = matrix_transpose(&m, 2, 3);
    assert_eq!(t.len(), 6);
    assert!((t[0] - 1.0).abs() < 1e-10);
    assert!((t[1] - 4.0).abs() < 1e-10);
    assert!((t[2] - 2.0).abs() < 1e-10);
    assert!((t[3] - 5.0).abs() < 1e-10);
    assert!((t[4] - 3.0).abs() < 1e-10);
    assert!((t[5] - 6.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_eigenvalues_2x2() {
    let m = vec![4.0, 1.0, 1.0, 3.0];
    let eigs = eigenvalues(&m, 2);
    assert_eq!(eigs.len(), 2);
    // Eigenvalues should be approximately 4.618 and 2.382
    assert!((eigs[0] - 4.618).abs() < 0.1);
    assert!((eigs[1] - 2.382).abs() < 0.1);
}

#[wasm_bindgen_test]
fn test_eigenvalues_diagonal() {
    let m = vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
    let eigs = eigenvalues(&m, 3);
    assert!((eigs[0] - 7.0).abs() < 0.1);
    assert!((eigs[1] - 5.0).abs() < 0.1);
    assert!((eigs[2] - 3.0).abs() < 0.1);
}

// ─── Laplacian Tests ─────────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_build_laplacian() {
    // Triangle graph
    let adj = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let lap = build_laplacian(&adj, 3);
    // Diagonal should be degree = 2
    assert!((lap[0] - 2.0).abs() < 1e-10);
    assert!((lap[4] - 2.0).abs() < 1e-10);
    assert!((lap[8] - 2.0).abs() < 1e-10);
    // Off-diagonal should be -1
    assert!((lap[1] - (-1.0)).abs() < 1e-10);
    assert!((lap[2] - (-1.0)).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_spectral_gap_connected() {
    // Connected triangle graph
    let adj = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let gap = algebraic_connectivity(&adj, 3);
    assert!(gap > 0.0, "Connected graph should have positive spectral gap");
}

#[wasm_bindgen_test]
fn test_spectral_gap_disconnected() {
    // Disconnected graph: two isolated vertices + one connected pair
    let adj = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let gap = algebraic_connectivity(&adj, 3);
    assert!(gap.abs() < 0.5, "Disconnected graph should have near-zero spectral gap");
}

#[wasm_bindgen_test]
fn test_count_components() {
    let adj = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let lap = build_laplacian(&adj, 3);
    let components = count_components(&lap, 3);
    assert_eq!(components, 2);
}

// ─── Heat Kernel Tests ───────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_heat_kernel_identity() {
    let lap = vec![0.0, 0.0, 0.0, 0.0];
    let hk = heat_kernel(&lap, 2, 1.0);
    // exp(-0) = I
    assert!((hk[0] - 1.0).abs() < 1e-8);
    assert!((hk[3] - 1.0).abs() < 1e-8);
}

#[wasm_bindgen_test]
fn test_heat_kernel_decay() {
    let adj = vec![0.0, 1.0, 1.0, 0.0];
    let lap = build_laplacian(&adj, 2);
    let hk_t0 = heat_kernel(&lap, 2, 0.0);
    let hk_t1 = heat_kernel(&lap, 2, 1.0);
    // Off-diagonal should grow (diffusion mixes)
    assert!(hk_t1[1].abs() > hk_t0[1].abs() || hk_t1[1] != 0.0);
}

#[wasm_bindgen_test]
fn test_harmonic_projection() {
    let adj = vec![0.0, 1.0, 1.0, 0.0];
    let lap = build_laplacian(&adj, 2);
    let signal = vec![1.0, 3.0];
    let proj = harmonic_projection(&lap, 2, &signal);
    assert_eq!(proj.len(), 2);
    // Should project to constant (mean)
    assert!((proj[0] - 2.0).abs() < 0.1);
    assert!((proj[1] - 2.0).abs() < 0.1);
}

// ─── Agent Lifecycle Tests ───────────────────────────────────────────

#[wasm_bindgen_test]
fn test_create_agent() {
    let json = create_agent("test", 3);
    assert!(json.contains("test"));
    assert!(json.contains("beliefs"));
}

#[wasm_bindgen_test]
fn test_agent_observe() {
    let state = create_agent("a1", 2);
    let obs = vec![0.8, 0.3];
    let updated = agent_observe(&state, &obs);
    assert!(!updated.is_empty());
    // Should contain updated beliefs
    assert!(updated.contains("observation_count"));
}

#[wasm_bindgen_test]
fn test_agent_predict() {
    let state = create_agent("a1", 3);
    let pred = agent_predict(&state);
    assert!(!pred.is_empty());
    let beliefs: Vec<f64> = serde_json::from_str(&pred).unwrap();
    assert_eq!(beliefs.len(), 3);
}

#[wasm_bindgen_test]
fn test_agent_act() {
    let state = create_agent("a1", 2);
    let actions = agent_act(&state);
    assert!(!actions.is_empty());
}

#[wasm_bindgen_test]
fn test_agent_conserve() {
    let state = create_agent("a1", 3);
    let obs = vec![0.9, 0.1, 0.5];
    let after_obs = agent_observe(&state, &obs);
    let after_conserve = agent_conserve(&after_obs);
    assert!(after_conserve.contains("conservation_integral"));
}

#[wasm_bindgen_test]
fn test_agent_cycle() {
    let state = create_agent("a1", 3);
    let obs = vec![0.7, 0.3, 0.5];
    let result = agent_cycle(&state, &obs);
    assert!(result.contains("observe"));
    assert!(result.contains("prediction"));
    assert!(result.contains("actions"));
    assert!(result.contains("conserve"));
}

#[wasm_bindgen_test]
fn test_agent_energy_depletion() {
    let mut state = create_agent("a1", 2);
    for _ in 0..1000 {
        let obs = vec![0.99, 0.01];
        state = agent_observe(&state, &obs);
        state = agent_conserve(&state);
    }
    let parsed: serde_json::Value = serde_json::from_str(&state).unwrap();
    assert!(parsed["energy"].as_f64().unwrap() < 1.0);
}

// ─── Conservation Tests ──────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_conservation_check() {
    let state = create_agent("a1", 3);
    let deviation = check_conservation(&state);
    assert!(deviation.abs() < 0.1, "Fresh agent should have near-zero conservation deviation");
}

#[wasm_bindgen_test]
fn test_conservation_after_observe() {
    let state = create_agent("a1", 3);
    let after = agent_observe(&state, &[0.9, 0.1, 0.3]);
    let after_conserve = agent_conserve(&after);
    let deviation = check_conservation(&after_conserve);
    assert!(deviation < 0.5, "Conservation enforcement should keep deviation small");
}

#[wasm_bindgen_test]
fn test_conservation_violation_multi() {
    let s1 = create_agent("a1", 2);
    let s2 = create_agent("a2", 2);
    let states = format!("[{},{}]", s1, s2);
    let violation = conservation_violation(&states);
    assert!(violation.abs() < 0.1);
}

// ─── Belief Utility Tests ────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_belief_entropy() {
    let state = create_agent("a1", 4);
    let ent = belief_entropy(&state);
    assert!(ent > 0.0, "Entropy of non-degenerate beliefs should be positive");
}

#[wasm_bindgen_test]
fn test_belief_kl_divergence() {
    let s1 = create_agent("a1", 3);
    let s2 = agent_observe(&s1, &[0.9, 0.1, 0.5]);
    let kl = belief_kl_divergence(&s1, &s2);
    assert!(kl >= 0.0, "KL divergence should be non-negative");
}

#[wasm_bindgen_test]
fn test_merge_agents() {
    let s1 = create_agent("a1", 3);
    let s2 = agent_observe(&create_agent("a2", 3), &[0.8, 0.2, 0.6]);
    let merged = merge_agents(&s1, &s2, 0.5);
    assert!(merged.contains("+"));
}

// ─── Cheeger & Spectral Tests ────────────────────────────────────────

#[wasm_bindgen_test]
fn test_cheeger_constant() {
    let adj = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let h = cheeger_constant(&adj, 3);
    assert!(h > 0.0, "Connected graph should have positive Cheeger constant");
}

// ─── Serialization Tests ─────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_matrix_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let json = matrix_to_json(&data, 2, 3);
    let parsed = json_to_matrix(&json);
    assert!(parsed.contains("rows"));
    assert!(parsed.contains("\"data\""));
}

#[wasm_bindgen_test]
fn test_version() {
    let v = version();
    assert!(!v.is_empty());
    assert!(v.contains('.'));
}

// ─── Utility Tests ───────────────────────────────────────────────────

#[wasm_bindgen_test]
fn test_softmax() {
    let v = vec![1.0, 2.0, 3.0];
    let s = softmax(&v);
    assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!(s[2] > s[1]);
    assert!(s[1] > s[0]);
}

#[wasm_bindgen_test]
fn test_vec_operations() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let sum = vec_add(&a, &b);
    assert_eq!(sum, vec![5.0, 7.0, 9.0]);

    let diff = vec_sub(&b, &a);
    assert_eq!(diff, vec![3.0, 3.0, 3.0]);

    let scaled = vec_scale(&a, 2.0);
    assert_eq!(scaled, vec![2.0, 4.0, 6.0]);

    let dot = vec_dot(&a, &b);
    assert!((dot - 32.0).abs() < 1e-10);

    let norm = vec_norm(&vec![3.0, 4.0]);
    assert!((norm - 5.0).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn test_clamp_vec() {
    let v = vec![-1.0, 0.5, 2.0];
    let clamped = clamp_vec(&v, 0.0, 1.0);
    assert_eq!(clamped, vec![0.0, 0.5, 1.0]);
}

#[wasm_bindgen_test]
fn test_set_panic_hook() {
    set_panic_hook();
    // Should not panic
}
