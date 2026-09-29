# Future latent dynamics

The current project predicts six typed transition properties for one candidate at a time. It does not learn a latent state transition or perform multi-step latent rollout.

A later experiment may learn state and action encoders `z_s, z_a`, predict `delta_z = f(z_s, z_a)`, and form `z_hat_next = z_s + delta_z`. That work requires representation-stability tests, multi-step rollout error curves, and direct comparison with the typed v2bis predictor.

The decision to begin latent-dynamics work waits for the completed `final-v001` result and an external computer-use evaluation. It is not part of the paused v2bis campaign.
