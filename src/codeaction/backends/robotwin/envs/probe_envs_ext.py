"""D9 load-path probe env: byte-identical behavior to upstream `stack_bowls_two`, resolved through
the `envs_ext` fallback of the generic loader. Exists ONLY to prove loader + audit wiring end-to-end
(local AST audit; remote boot smoke); never a benchmark task, never listed in any registry."""
from envs.stack_bowls_two import stack_bowls_two


class probe_envs_ext(stack_bowls_two):
    pass
