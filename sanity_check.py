import jax

try:
    key = jax.random.PRNGKey(0)
    key, *subs = jax.random.split(key, 10)
    print("All Good!")
except Exception as e:
    print(e)
