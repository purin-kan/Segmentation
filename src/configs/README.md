# Configs

Per-method / per-experiment configuration files (paths, hyperparameters, split seeds).

Convention: one YAML file per method, named after its `src/s3_methods/<family>/<name>.py`
module, e.g. `m5_deep_learning_d_unet.yaml`. No configs exist yet — methods are still stubs
(see `docs/implementation_plan.md`).
