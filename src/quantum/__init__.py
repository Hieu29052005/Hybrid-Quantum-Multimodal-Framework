from .quantum_fusion import QuantumFusionTensor, QuantumFusionAttention, QuantumFusionInterference
from .quantum_attention import QuantumMultiHeadAttention
from .shared_quantum_space import SharedQuantumSpace
from .circuits import angle_encoding, amplitude_encoding, iqp_encoding, strongly_entangling_layer, basic_entangler_layer
from .encoding import ENCODINGS, get_encoding, max_features_for, estimate_encoding_cost
from .noise import add_depolarizing_noise, create_noisy_device, compute_fidelity
from .entanglement import (
    CrossModalEntanglementAnalyzer,
    partial_trace,
    von_neumann_entropy,
    mutual_information,
    meyer_wallach_from_rho,
    make_dm_circuit,
)
from .ansatz_metrics import (
    expressibility,
    entangling_capability,
    profile_ansatz,
    ANSATZ_REGISTRY,
)
from .noise_wrapper import (
    DepolarizingNoiseWrapper,
    ResidualQuantumMitigation,
    apply_component_noise,
    restore_component_noise,
)
