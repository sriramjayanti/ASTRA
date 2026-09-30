import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator

p_gen = PayloadGenerator()
f_gen = FrameGenerator()
fec_gen = FECGenerator()
int_gen = InterleaverGenerator()
mod_gen = ModulationGenerator()
chan_gen = ChannelGenerator()

p = p_gen.generate(payload_type='counter', bit_length=512, seed=999)
f = f_gen.generate(p)
fec = fec_gen.encode(f, scheme='none')
inter = int_gen.interleave(fec, scheme='none')
m = mod_gen.modulate(inter, modulation_type='qpsk', sample_rate=192000.0, symbol_rate=9600.0, samples_per_symbol=20)
c = chan_gen.apply(m, overrides={'snr_db': 15.0, 'cfo_hz': 250.0})
print(f"Success! Clean IQ len: {len(m.clean_iq)}, Impaired IQ len: {len(c.impaired_iq)}")
