import julia
import numpy as np
jl = julia.Julia(compiled_modules=False)
jl.eval("using Flux")
jl.eval("using BSON")
jl.eval("using CUDA")
load_jl = jl.eval("BSON.load")

m = load_jl("checkpoints/model_10.bson")
actor = m["actor"]
state = np.zeros(22, dtype=np.float32)
action = actor(state)
print(action)
