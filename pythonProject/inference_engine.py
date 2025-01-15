class InferenceEngine:
    def __init__(self, network):
        self.network = network

    def enumerate_all(self, variables, evidence):
        """Enumerate all variables recursively for inference."""
        if not variables:
            return 1.0

        first, rest = variables[0], variables[1:]

        if first in evidence:
            prob = self.network.get_probability(first, evidence[first], evidence)
            return prob * self.enumerate_all(rest, evidence)
        else:
            total = 0
            for value in [True, False]:
                extended_evidence = evidence.copy()
                extended_evidence[first] = value
                prob = self.network.get_probability(first, value, extended_evidence)
                total += prob * self.enumerate_all(rest, extended_evidence)
            return total

    def query(self, query_var, evidence):
        """Calculate the probability distribution of the query variable."""
        variables = list(self.network.nodes.keys())
        probabilities = {}

        for value in [True, False]:
            extended_evidence = evidence.copy()
            extended_evidence[query_var] = value
            probabilities[value] = self.enumerate_all(variables, extended_evidence)

        # Normalize probabilities
        total = sum(probabilities.values())
        for value in probabilities:
            probabilities[value] /= total

        return probabilities
