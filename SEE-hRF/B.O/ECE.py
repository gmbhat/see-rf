import numpy as np


def calculate_confidence_max_prob(probabilities):
    probabilities = np.array(probabilities)  # convert list of lists to ndarray
    return np.max(probabilities, axis=1).tolist()



def ECE_computation(confidences, corrects, num_bins=10):
    # Divide the confidence interval [0,1] into equal-width bins
    bin_bounds = np.linspace(0, 1, num_bins + 1)  # list of bin edges
    ECE = 0.0

    # Loop through each bin
    for i in range(num_bins):
        b_0 = bin_bounds[i]
        b_1 = bin_bounds[i + 1]

        # Select indices of samples whose confidence falls into the bin
        samples_indices = [j for j, c in enumerate(confidences) if b_0 <= c < b_1]

        if len(samples_indices) == 0:
            continue  # skip empty bins

        acc_mean = np.mean([corrects[j] for j in samples_indices])
        conf_mean = np.mean([confidences[j] for j in samples_indices])

        weight = len(samples_indices) / len(confidences)
        ECE += weight * abs(acc_mean - conf_mean)

    return round(ECE, 2)

