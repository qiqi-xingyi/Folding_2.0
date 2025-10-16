from qangle import FourierAngleEnergyBuilder, LocalStatevectorEstimator, AngleCircuit

def test_smoke():
    builder = FourierAngleEnergyBuilder(n_angles=2)
    builder.add_single(0, a=[0.8], b=[-0.1])
    builder.add_single(1, a=[0.5], b=[0.0])
    builder.add_coupling(0,1, weight=0.6)
    H = builder.to_sparse_pauli(n_qubits=2)
    circ = AngleCircuit(n_angles=2)
    est = LocalStatevectorEstimator()
    E = est.expectation(circ.build([0.1, -1.2]), H)
    assert isinstance(E, float)
