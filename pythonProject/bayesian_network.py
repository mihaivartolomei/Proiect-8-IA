import json

class BayesianNetwork:
    def __init__(self, structure_file):
        """Initialize the Bayesian Network from a file."""
        self.nodes = {}
        self.probabilities = {}
        self.load_structure(structure_file)

    def load_structure(self, structure_file):
        """Load network structure and probabilities from a JSON file."""
        with open(structure_file, 'r') as f:
            data = json.load(f)

        for node, details in data.items():
            self.nodes[node] = details['parents']
            self.probabilities[node] = details['probabilities']

    def get_parents(self, node):
        return self.nodes.get(node, [])

    def get_probability(self, node, value, evidence):
        """Retrieve the conditional probability for a node."""
        parents = self.get_parents(node)
        if not parents:
            # Node has no parents; return marginal probability
            return self.probabilities[node][str(value)]

        # Build the key for conditional probability lookup
        if len(parents) == 1:
            # Special case for single parent: tuple with one element
            evidence_key = f"({str(evidence[parents[0]])},)"
        else:
            evidence_key = "(" + ", ".join([str(evidence[parent]) for parent in parents]) + ")"

        try:
            return self.probabilities[node][evidence_key][str(value)]
        except KeyError:
            print(f"Missing probability for node '{node}' with evidence '{evidence_key}' and value '{value}'.")
            raise
