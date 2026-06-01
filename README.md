# Lau Math WASM

> Browser/edge/WebAssembly implementation of Lau math — agent computation, spectral analysis, and conservation laws running natively in the browser.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![WASM](https://img.shields.io/badge/wasm-wasm_pack-blue.svg)](https://rustwasm.github.io/wasm-pack/)

## What It Does

This is the **browser and edge runtime** for Lau math — compiled to WebAssembly via Rust + wasm-bindgen. It handles:

- **Client-side agent computation** — create, observe, predict, update, act, conserve
- **Edge inference** — run agent logic on Cloudflare Workers, Deno Deploy, or any WASM runtime
- **Browser-based fleet visualization** — spectral gaps, belief states, conservation monitoring
- **Serverless agent functions** — stateless agent lifecycle for serverless architectures

## Architecture

```
┌─────────────────────────────────────────────┐
│                Browser / Edge               │
│  ┌─────────────────────────────────────┐    │
│  │         lau-math-wasm (WASM)        │    │
│  │  • Matrix ops (multiply, inverse)   │    │
│  │  • Eigenvalue computation           │    │
│  │  • Laplacian & spectral gap         │    │
│  │  • Heat kernel & harmonic proj      │    │
│  │  • Agent lifecycle (O→P→U→A→C)     │    │
│  │  • Conservation monitoring          │    │
│  └─────────────────────────────────────┘    │
│  ┌──────────────┐  ┌───────────────────┐    │
│  │  JS/TS API   │  │  React/Vue/Svelte │    │
│  │  wasm-bindgen│  │  visualization    │    │
│  └──────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────┘
```

## API Reference

### Matrix Operations

```javascript
import init, { 
    matrix_multiply, matrix_inverse, eigenvalues, 
    matrix_transpose, matrix_to_json 
} from './pkg/lau_math_wasm.js';

await init();

// Multiply 2x2 matrices (flat row-major arrays)
const result = matrix_multiply([1,2,3,4], 2, 2, [5,6,7,8], 2, 2);
// → [19, 22, 43, 50]

// Inverse
const inv = matrix_inverse([4,3,3,2], 2);

// Eigenvalues (up to 8x8)
const eigs = eigenvalues([4,1,1,3], 2);
// → [4.618..., 2.382...]
```

### Graph & Laplacian

```javascript
import { build_laplacian, spectral_gap, algebraic_connectivity, 
         heat_kernel, harmonic_projection, cheeger_constant } from './pkg/lau_math_wasm.js';

// Triangle graph adjacency
const adj = [0,1,1, 1,0,1, 1,1,0];
const lap = build_laplacian(adj, 3);
// → [[2,-1,-1], [-1,2,-1], [-1,-1,2]]

const gap = spectral_gap(lap, 3);           // Fiedler value
const conn = algebraic_connectivity(adj, 3); // Same thing, one call
const hk = heat_kernel(lap, 3, 1.0);        // Heat kernel H(t)
const cheeger = cheeger_constant(adj, 3);    // Edge expansion approx
```

### Agent Lifecycle

The agent follows the Lau conservation cycle: **observe → predict → update → act → conserve**.

```javascript
import { create_agent, agent_observe, agent_predict, 
         agent_update, agent_act, agent_conserve, agent_cycle } from './pkg/lau_math_wasm.js';

// Create agent with 4-dimensional belief state
let state = create_agent("agent-1", 4);
// → JSON string with beliefs, confidence, energy, etc.

// Single lifecycle step (all 5 phases)
const result = JSON.parse(agent_cycle(state, [0.7, 0.3, 0.5, 0.6]));
state = result.conserve;

// Or run individual phases:
state = agent_observe(state, [0.7, 0.3, 0.5, 0.6]);
const prediction = agent_predict(state);
state = agent_update(state);
const actions = agent_act(state);
state = agent_conserve(state);
```

### Conservation Monitoring

```javascript
import { check_conservation, conservation_violation, 
         belief_entropy, belief_kl_divergence } from './pkg/lau_math_wasm.js';

const deviation = check_conservation(state);  // Deviation from conserved quantity
const entropy = belief_entropy(state);         // Belief state entropy
```

### Multi-Agent Operations

```javascript
import { merge_agents } from './pkg/lau_math_wasm.js';

// Distributed consensus merge
const merged = merge_agents(stateA, stateB, 0.5); // 50/50 weight
```

### Utilities

```javascript
import { softmax, clamp_vec, vec_add, vec_sub, 
         vec_scale, vec_dot, vec_norm, random, now_ms } from './pkg/lau_math_wasm.js';

const probs = softmax([1.0, 2.0, 3.0]);  // → normalized probabilities
const dist = vec_norm(vec_sub(a, b));      // Euclidean distance
```

## Building

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack
```

### Build for Browser

```bash
wasm-pack build --target web
```

This generates the `pkg/` directory with:
- `lau_math_wasm.js` — JS bindings
- `lau_math_wasm_bg.wasm` — WASM binary
- `lau_math_wasm.d.ts` — TypeScript definitions

### Build for Bundler (webpack, vite, etc.)

```bash
wasm-pack build --target bundler
```

### Build for Node.js

```bash
wasm-pack build --target nodejs
```

## Browser Demo

### Quick Start

```bash
# Build
wasm-pack build --target web

# Serve (any static server works)
cd www
python3 -m http.server 8080
# or
npx serve .
```

Open `http://localhost:8080` — the demo page lets you:

1. **Initialize** an agent with 4-dimensional beliefs
2. **Feed observations** via sliders
3. **Watch** the conservation law in real-time
4. **Visualize** spectral gap of the agent's interaction graph
5. **Track** learning trajectory (prediction error over time)

### What You'll See

- **Agent State** — bar chart of beliefs (green) and confidence (orange)
- **Conservation Monitor** — real-time deviation from the conserved quantity
- **Spectral Gap** — eigenvalues of the Laplacian for the interaction graph
- **Learning Trajectory** — prediction error converging over time

## Cloudflare Workers Deployment

### Setup

```bash
# Build for bundler target
wasm-pack build --target bundler

# Create Workers project
mkdir lau-worker && cd lau-worker
npm init -y
npm install wrangler --save-dev
```

### wrangler.toml

```toml
name = "lau-math-agent"
main = "src/index.js"
compatibility_date = "2024-01-01"

[build]
command = "cp ../pkg/lau_math_wasm_bg.wasm ./src/ && cp ../pkg/lau_math_wasm.js ./src/"
```

### src/index.js

```javascript
import init, { 
    create_agent, agent_cycle, check_conservation, version 
} from './lau_math_wasm.js';

let wasmInitialized = false;

export default {
    async fetch(request) {
        if (!wasmInitialized) {
            await init();
            wasmInitialized = true;
        }

        const url = new URL(request.url);

        if (url.pathname === '/cycle' && request.method === 'POST') {
            const { state, observation } = await request.json();
            const result = agent_cycle(state, observation);
            return new Response(result, {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        if (url.pathname === '/create') {
            const { id, dim } = await request.json();
            const state = create_agent(id || 'worker-agent', dim || 4);
            return new Response(state, {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        if (url.pathname === '/health') {
            return new Response(JSON.stringify({ 
                status: 'ok', 
                version: version(),
                runtime: 'cloudflare-workers'
            }), {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        return new Response('Lau Math WASM — use /create or /cycle endpoints', { status: 404 });
    }
};
```

### Deploy

```bash
npx wrangler deploy
```

### Deno Deploy

```typescript
// deno.ts
import init, { create_agent, agent_cycle } from './pkg/lau_math_wasm.js';

await init();

Deno.serve(async (req: Request) => {
    const url = new URL(req.url);
    
    if (url.pathname === '/cycle') {
        const { state, observation } = await req.json();
        const result = agent_cycle(state, new Float64Array(observation));
        return new Response(result, {
            headers: { 'Content-Type': 'application/json' }
        });
    }
    
    return new Response('Lau Math WASM on Deno');
});
```

## Test Suite

```bash
# Run tests in headless browser (requires wasm-pack)
wasm-pack test --chrome --headless

# Or with Firefox
wasm-pack test --firefox --headless
```

30+ tests covering:
- Matrix operations (multiply, inverse, transpose, eigenvalues)
- Laplacian construction and spectral analysis
- Heat kernel computation
- Agent lifecycle (all 5 phases)
- Conservation law enforcement
- Belief state utilities (entropy, KL divergence)
- Multi-agent merging
- Serialization roundtrips

## Performance

Typical benchmarks (Chrome, M1 Mac):

| Operation | Size | Time |
|-----------|------|------|
| Matrix multiply | 4×4 | < 1μs |
| Eigenvalues | 4×4 | ~50μs |
| Agent cycle (full) | dim=8 | ~5μs |
| Conservation check | dim=8 | < 1μs |
| Laplacian build | 8 nodes | < 1μs |

## Project Structure

```
lau-math-wasm/
├── src/
│   ├── lib.rs          # Core: matrices, Laplacian, agents, conservation
│   └── utils.rs        # WASM utilities: logging, random, vector math
├── tests/
│   └── web.rs          # 30+ browser tests
├── www/
│   └── index.html      # Interactive demo page
├── pkg/                # Generated by wasm-pack (not in repo)
├── Cargo.toml
└── README.md
```

## License

MIT
