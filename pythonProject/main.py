from bayesian_network import BayesianNetwork
from inference_engine import InferenceEngine

if __name__ == "__main__":
    print("Welcome to the Bayesian Network Inference Tool!")

    # Load Bayesian network from file
    structure_file = input("Enter the file path for the Bayesian network structure: ")
    bn = BayesianNetwork(structure_file)

    # Create inference engine
    engine = InferenceEngine(bn)

    while True:
        print("\nOptions:")
        print("1. Query a variable")
        print("2. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            query_var = input("Enter the query variable: ")
            evidence = {}
            print("Enter evidence variables (leave empty to finish):")

            while True:
                evidence_var = input("Evidence variable: ")
                if not evidence_var:
                    break
                value = input(f"Value for {evidence_var} (True/False): ").lower() == "true"
                evidence[evidence_var] = value

            probabilities = engine.query(query_var, evidence)
            print(f"\nProbability distribution for {query_var}:")
            for value, prob in probabilities.items():
                print(f"  {value}: {prob:.4f}")

        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
