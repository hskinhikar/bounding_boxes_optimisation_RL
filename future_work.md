# Future Works

- **Current Limitations in Training**:
  - The existing reinforcement learning script is technically capable of training the model. However, due to limited resources, it has not been feasible to continue training through the entire dataset, let alone multiple epochs - reinforcement learning is an incredibly time-consuming and resource-intensive process so this is expected.
  
- **Parallelized Environments**:
  - It is advisable to explore the implementation of parallelized environments to allow the model to train on multiple images simultaneously. This approach could significantly reduce training time.
  - However, this comes with challenges, such as ensuring the correct format of observations in vectorized environments. This has been a source of difficulty, particularly during experiments with parallel environments.

- **Visualization Considerations**:
  - The current visualization code is not designed for scenarios where multiple images are being trained on simultaneously. Turning off visualization during parallelized training may be necessary to conserve resources and avoid unnecessary complexity.

- **Resource Constraints**:
  - Training on a single image is already resource-intensive and time-consuming. Extending this to multiple images could overwhelm the available hardware, potentially causing crashes. Utilizing a virtual CPU/GPU or more powerful hardware is recommended.

- **Inference without Ground Truth**:
  - Once the model is trained, a separate script will be needed to test the model’s inference capabilities. If the model has been trained with ground truth as part of its observation space, it may be technically challenging to modify the model to produce inferences without this data. Despite these challenges, enabling inference without ground truth is the ultimate goal of reinforcement learning and is feasible with the right approach.

- **Reward and Evaluation Visualization**:
  - Implement a script to display reward and cumulative reward curves, along with other evaluation metrics, to better assess model performance during and after training.
