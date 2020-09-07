swimmer_rnd = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_vf_coef":0.5, "rnd_start":2e+3, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":64}

swimmer_icm = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_rew_integration":0.1, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":32, "beta":0.2, "policy_weight":1}

swimmer_ppo = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0}


inverted_pendulum_ppo = {"hidden_size":64, "lr": 0.001, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "max_grad_norm":5, "nstep":256,
                            "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0}

inverted_pendulum_icm = {"hidden_size":64, "lr": 0.001, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_rew_integration":0.1, "max_grad_norm":5, "nstep":256,
                            "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.001, "int_hidden_size":32, "beta":0.2, "policy_weight":1}

inverted_pendulum_rnd = {"hidden_size":64, "lr": 0.001, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_vf_coef":0.5, "rnd_start":2e+3, "max_grad_norm":5, "nstep":256,
                            "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.001, "int_hidden_size":64}


inverted_double_pendulum_ppo = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "max_grad_norm":5, "nstep":2048,
                            "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0}

inverted_double_pendulum_icm = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_rew_integration":0.1, "max_grad_norm":5, "nstep":2048,
                            "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":32, "beta":0.2, "policy_weight":1}

inverted_double_pendulum_rnd = {"hidden_size":64, "lr": 0.0003, "gamma":0.999, "gae_lam": 0.95, "vf_coef":1, "int_vf_coef":0.5, "rnd_start":2e+3, "max_grad_norm":5, "nstep":2048,
                            "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":32}


reacher_ppo = {"hidden_size":64, "lr": 0.001, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "max_grad_norm":5, "nstep":256,
                        "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0}

reacher_icm = {"hidden_size":64, "lr": 0.001, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "int_rew_integration":0.1, "max_grad_norm":5, "nstep":256,
                        "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.001, "int_hidden_size":32, "beta":0.2, "policy_weight":1}

reacher_rnd = {"hidden_size":64, "lr": 0.001, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "int_vf_coef":0.5, "rnd_start":2e+3, "max_grad_norm":5, "nstep":256,
                        "batch_size":64, "n_epochs":4, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.001, "int_hidden_size":32}


hopper_rnd = {"hidden_size":64, "lr": 0.0003, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "int_vf_coef":0.5, "rnd_start":2e+3, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":128}

hopper_icm = {"hidden_size":64, "lr": 0.0003, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "int_rew_integration":0.1, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0, "int_lr":0.0003, "int_hidden_size":32, "beta":0.2, "policy_weight":0.1}

hopper_ppo = {"hidden_size":64, "lr": 0.0003, "gamma":0.99, "gae_lam": 0.95, "vf_coef":1, "max_grad_norm":5, "nstep":2048,
                "batch_size":64, "n_epochs":10, "clip_range":0.2, "ent_coef":0.0}