use wasm_bindgen::prelude::*;

/// Log to browser console
#[wasm_bindgen]
pub fn console_log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}

/// Log warning to browser console
#[wasm_bindgen]
pub fn console_warn(msg: &str) {
    web_sys::console::warn_1(&msg.into());
}

/// Get a random f64 in [0, 1) — delegates to Math.random()
#[wasm_bindgen]
pub fn random() -> f64 {
    js_sys::Math::random()
}

/// Get high-resolution timestamp in milliseconds
#[wasm_bindgen]
pub fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// Generate random matrix (for testing)
#[wasm_bindgen]
pub fn random_matrix(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|_| js_sys::Math::random()).collect()
}

/// Generate symmetric random matrix (for adjacency/Laplacian testing)
#[wasm_bindgen]
pub fn random_symmetric_matrix(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = js_sys::Math::random();
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    m
}

/// Softmax over a vector
#[wasm_bindgen]
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Clamp values to [min, max]
#[wasm_bindgen]
pub fn clamp_vec(values: &[f64], min: f64, max: f64) -> Vec<f64> {
    values.iter().map(|v| v.max(min).min(max)).collect()
}

/// Element-wise vector operations
#[wasm_bindgen]
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[wasm_bindgen]
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[wasm_bindgen]
pub fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

#[wasm_bindgen]
pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[wasm_bindgen]
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Set panic hook for better error messages in browser
#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}
