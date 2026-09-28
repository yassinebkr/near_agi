# Future latent dynamics (not Milestone 1)

Only after a one-step typed predictor demonstrates held-out utility should the project learn state/action encoders `z_s, z_a`, predict `delta_z = f(z_s, z_a)`, and form `z_hat_next = z_s + delta_z`. Training would require representation-stability checks, multi-step rollout error curves, and comparisons against the typed predictor. No latent rollout is implemented here.

