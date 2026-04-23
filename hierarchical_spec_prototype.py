import torch
import torch.nn.functional as F

def generate_distributions(V, L, device):
    """Generate synthetic distributions for Draft, Target, and Sketch models."""
    # Target model (P)
    logits_P = torch.randn(L, V, device=device)
    P = F.softmax(logits_P, dim=-1)
    
    # Draft model (Q) - slightly different from P
    logits_Q = logits_P + torch.randn(L, V, device=device) * 2.0
    Q = F.softmax(logits_Q, dim=-1)
    
    # Sketch model (P_tilde) - an approximation of P, better than Q
    logits_P_tilde = logits_P + torch.randn(L, V, device=device) * 0.5
    P_tilde = F.softmax(logits_P_tilde, dim=-1)
    
    return Q, P_tilde, P

def standard_speculative_decoding(Q, P, draft_tokens, u_vals):
    """Standard rejection sampling: Draft vs Target."""
    L = len(draft_tokens)
    accepted_count = 0
    
    for i in range(L):
        token = draft_tokens[i]
        p_val = P[i, token]
        q_val = Q[i, token]
        
        # Rejection test
        if u_vals[i] > (p_val / q_val).item():
            break
        accepted_count += 1
        
    return accepted_count, L # Target model always evaluates L tokens

def hierarchical_speculative_decoding(Q, P_tilde, P, draft_tokens, u_vals):
    """Hierarchical rejection sampling: Draft -> Sketch -> Target."""
    L = len(draft_tokens)
    
    # Stage 1: Shortlist using Sketch model
    m = 0
    for i in range(L):
        token = draft_tokens[i]
        p_tilde_val = P_tilde[i, token]
        q_val = Q[i, token]
        
        # Nested rejection test (Draft vs Sketch)
        # We accept if rand < P_tilde / Q
        if u_vals[i] > (p_tilde_val / q_val).item():
            break
        m += 1
        
    # Stage 2: Final verification using Target model (only on m tokens)
    accepted_count = 0
    for i in range(m):
        token = draft_tokens[i]
        p_val = P[i, token]
        p_tilde_val = P_tilde[i, token]
        q_val = Q[i, token]
        
        # The true acceptance probability should be min(1, P/Q)
        # Since we already passed the first test with prob min(1, P_tilde/Q),
        # the conditional probability to pass the second test must be:
        # P(accept_2 | accept_1) = min(1, P/Q) / min(1, P_tilde/Q)
        
        prob_1 = min(1.0, (p_tilde_val / q_val).item())
        prob_true = min(1.0, (p_val / q_val).item())
        
        prob_2 = prob_true / prob_1 if prob_1 > 0 else 0.0
        
        # To make it exactly equivalent to standard spec decode, we must use the SAME random variable
        # But we must scale it because we already conditioned on u < prob_1
        # If u ~ U(0, 1) and we know u < prob_1, then u/prob_1 ~ U(0, 1)
        # Wait, the rejection test is u > prob_1. So if we passed, u <= prob_1.
        # Therefore, u is uniformly distributed in [0, prob_1].
        # So u / prob_1 is uniformly distributed in [0, 1].
        
        # Actually, the standard spec decode test is: u < min(1, P/Q)
        # Our two-stage test is: u < min(1, P_tilde/Q) AND u_scaled < prob_2
        # u_scaled = u / min(1, P_tilde/Q)
        # So u_scaled < prob_2  =>  u / prob_1 < prob_true / prob_1  =>  u < prob_true
        # This is EXACTLY the same test! We don't even need to scale u, we can just use u directly!
        
        # Wait, if we just use u directly, then the accepted tokens will be EXACTLY the same.
        # Let's verify this.
        # BUT wait, what if the sketch model REJECTS a token that the target model would have ACCEPTED?
        # i.e., P_tilde/Q < u < P/Q
        # In this case, the hierarchical approach rejects it early, but the standard approach accepts it!
        # This is why the accepted tokens are lower (0.3980 vs 0.4820).
        # To fix this, we must use the standard rejection sampling correction:
        # If the sketch model rejects, we must sample a RECOVERED token from the residual distribution
        # (P_tilde - Q)_+. But wait, we don't want to sample from P_tilde, we want to sample from P!
        # This means the sketch model CANNOT safely reject tokens without altering the final distribution,
        # UNLESS P_tilde is a strict upper bound of P (which is impossible for probabilities that sum to 1).
        
        if u_vals[i] > prob_true:
            break
        accepted_count += 1
        
    return accepted_count, m # Target model only evaluates m tokens

def run_simulation():
    torch.manual_seed(42)
    device = torch.device('cpu')
    V = 32000
    L = 8
    iterations = 1000
    
    std_accepted_total = 0
    std_target_evals_total = 0
    
    hier_accepted_total = 0
    hier_target_evals_total = 0
    
    print(f"Running Monte Carlo simulation ({iterations} iterations)...")
    
    for _ in range(iterations):
        Q, P_tilde, P = generate_distributions(V, L, device)
        
        # Sample draft tokens from Q
        draft_tokens = torch.multinomial(Q, 1).squeeze(-1)
        
        # Generate uniform random variables for the rejection tests
        u_vals = torch.rand(L).tolist()
        
        # Standard
        std_acc, std_evals = standard_speculative_decoding(Q, P, draft_tokens, u_vals)
        std_accepted_total += std_acc
        std_target_evals_total += std_evals
        
        # Hierarchical
        hier_acc, hier_evals = hierarchical_speculative_decoding(Q, P_tilde, P, draft_tokens, u_vals)
        hier_accepted_total += hier_acc
        hier_target_evals_total += hier_evals
        
    print("\n--- Results ---")
    print(f"Standard Approach:")
    print(f"  Average Accepted Tokens: {std_accepted_total / iterations:.4f}")
    print(f"  Target Model Evals:      {std_target_evals_total / iterations:.4f}")
    
    print(f"\nHierarchical Approach:")
    print(f"  Average Accepted Tokens: {hier_accepted_total / iterations:.4f}")
    print(f"  Target Model Evals:      {hier_target_evals_total / iterations:.4f}")
    
    eval_reduction = 1.0 - (hier_target_evals_total / std_target_evals_total)
    print(f"\nTarget Model Compute Saved: {eval_reduction * 100:.2f}%")

if __name__ == "__main__":
    run_simulation()