import jax
import jax.numpy as jnp
try:
    import jax.scipy.signal
    print("jax.scipy.signal imported")
    a = jnp.array([1, 2, 3])
    b = jnp.array([0, 1, 0.5])
    c = jax.scipy.signal.correlate(a, b, mode='full')
    print("correlate result:", c)
except Exception as e:
    print("Error:", e)
