import json
import numpy as np
import strawberryfields as sf
from strawberryfields.tdm import borealis_gbs, get_mode_indices
from strawberryfields.ops import Sgate, Rgate, BSgate, MeasureFock


"""
Configure Xanadu Cloud API key:
$ xcc config set REFRESH_TOKEN "Xanadu Cloud API key goes here"
$ xcc ping
Successfully connected to the Xanadu Cloud.
"""


class Borealis:
    def __init__(self):
        self.eng = sf.RemoteEngine("simulon_gaussian")
        self.device = self.eng.device

    def get_samples(self):
        gate_args_list = borealis_gbs(self.device, modes=216, squeezing="high")
        delays = [1, 6, 36]
        n, N = get_mode_indices(delays)

        prog = sf.TDMProgram(N)

        with prog.context(*gate_args_list) as (p, q):
            Sgate(p[0]) | q[n[0]]
            for i in range(len(delays)):
                Rgate(p[2 * i + 1]) | q[n[i]]
                BSgate(p[2 * i + 2], np.pi / 2) | (q[n[i + 1]], q[n[i]])
            MeasureFock() | q[0]

        shots = 10_000
        results = self.eng.run(prog, shots=shots, crop=True)
        print(results.samples)

        return results.samples


borealis = Borealis()

for i in range(10):
    samples = borealis.get_samples()
    with open(f"quantum_noise/{i}.json", "w") as f:
        json.dump(samples.tolist(), f)
