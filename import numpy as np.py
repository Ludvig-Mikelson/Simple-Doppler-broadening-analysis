import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import glob
import os

# Load data using pandas
names = glob.glob("C:/Users/Deloading/Desktop/Julia/LC/CsD2_Spektrs/*.txt")

for file_path in names:
    data = pd.read_csv(file_path, sep='\t', header=0)  # Treat first row as header, adjust delimiter if needed

    file_name = os.path.splitext(os.path.basename(file_path))[0]

    time_data = data.iloc[:, 0].to_numpy()
    y = data.iloc[:, 1].to_numpy()

    # Fit GMM
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data)
    
    means = gmm.means_
    covariances = gmm.covariances_

    # Plotting
    plt.figure()  # Create a new figure outside the loop
for i in range(3):
    mean = means[i]
    cov = covariances[i]
    std_dev = np.sqrt(np.diag(cov))  # Extract diagonal elements (variances) and take square root
    normal_distribution = (
        1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-(time_data - mean)**2 / (2 * std_dev**2))
    )
    plt.plot(time_data, normal_distribution, label=f'Component {i+1}')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Separate Normal Distributions')
plt.show()