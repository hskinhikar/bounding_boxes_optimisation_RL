import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from torchvision.models import ResNet18_Weights
from torchvision import models
import torch as th
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import os
import json
from gen_predicted_grndreality_polygons import process_directory
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon

class PolygonEnv(gym.Env):

    """
    A custom Gym environment designed for reinforcement learning, where:

    - The environment observes the vertices of the initially predicted polygon and the associated image. 

    - It adjusts the vertices of the initial polygon to maximize rewards according to the defined reward policy.

    Reward Policy:

    - Positive reward for achieving a high IoU (Intersection over Union) between the predicted polygon and the ground truth polygon.
      This incentivizes the predicted polygon to closely align with the ground truth, maximizing overlap.

    - Negative reward for any areas of the predicted polygon that do not overlap with the ground truth polygon.
      This discourages the polygon from incorrectly covering areas outside the ground truth.

    - Negative reward for any actions that cause the edges of the polygon to intersect or overlap incorrectly.
      This prevents the formation of invalid polygon shapes where IoU cannot be accurately calculated.

    - Negative reward for any actions that move the vertices of the polygon outside the image bounds.
      This penalty is scaled by the IoU area to ensure it does not overpower other rewards and penalties.
    """

    def __init__(self, initial_polygon, ground_truth_polygon, image):
        super(PolygonEnv, self).__init__()
        self.initial_polygon = initial_polygon
        self.current_polygon = initial_polygon
        self.ground_truth_polygon = ground_truth_polygon
        self.image = image

        # action space defines all actions that can be taken by the agent - in this case to move vertice
        self.action_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        # defines formats/limits of observations that the environment returns to the agent
        # info about image and vertices are returned to agent after every episode so that it can improve its policy
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, self.image.size[1], self.image.size[0]), dtype=np.uint8),
            "vertices": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        """
        Initialise the environment at the start of each episode
        Sets up the initial state of the environment and returns the initial observations
        """
        super().reset(seed=seed)
        self.current_polygon = self.initial_polygon
        image_array = np.array(self.image).transpose(2, 0, 1)  # (channels, height, width)

        observations = {
            "image": image_array,
            "vertices": np.array(self.current_polygon.exterior.coords)[:-1].flatten()
        }
        # return observations as dictionaries and not single arrays to attempt to satisfy sb3 format
        info = {}
        return observations, info

    def step(self, action):
        """
        Takes an action and returns next observation as a dictionary, along with reward, done, truncated and info
        returns numpy array, reward, done, truncated and info
        """
        vertices = np.array(self.current_polygon.exterior.coords)[:-1]# Current vertices of polygon
        adjusted_vertices = vertices + action.reshape(-1, 2)# Adjust vertices based on action

        new_polygon = Polygon(adjusted_vertices)# Create new polygon with adjusted vertices
        new_polygon = orient(new_polygon, sign=1.0)

        # Check if the adjusted vertices are within the image bounds
        img_height, img_width = self.image.size[1], self.image.size[0]
        
        # Define the image boundary polygon
        image_boundary_polygon = Polygon([(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)])
        

        if not new_polygon.is_valid:
            reward = -1.0#penalise if new polygon is not valid, may be useful to experiment with the stregth of penalty
        else:
            self.current_polygon = new_polygon
            iou = self.calculate_iou(self.current_polygon, self.ground_truth_polygon)
            non_overlap_penalty = self.calculate_non_overlap_penalty(self.current_polygon, self.ground_truth_polygon)
            # Calculate the area of the predicted polygon that is outside the image bounds
            outside_area = max(new_polygon.difference(image_boundary_polygon).area, 0)
            # Scale the penalty by IoU
            outside_penalty = -iou * outside_area 
            reward = iou - (1 - iou) - non_overlap_penalty + outside_penalty # Compile reward

        done = self.is_done(self.current_polygon, self.ground_truth_polygon)
        info = {}
        truncated = False

        obs = {
            "image": np.array(self.image).transpose(2, 0, 1),  # convert image to numpy array with shape (channels, height, width)
            "vertices": np.array(self.current_polygon.exterior.coords)[:-1].flatten() # flatten polygons vertices to 1D
        }

        return obs, reward, done, truncated, info

    def calculate_iou(self, poly1, poly2):
        # calculates Intersection over Union - metric for measuring overlap between the initial and ground truth polygons
        if not poly1.is_valid:
            return 0
        if not poly2.is_valid:
            return 0
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union

    def calculate_non_overlap_penalty(self, predicted_polygon, ground_truth_polygon):
        # gives penalty for amount of area of the initial polygon that doesn't overlap with ground truth
        intersection_area = predicted_polygon.intersection(ground_truth_polygon).area
        predicted_area = predicted_polygon.area
        non_overlap_area = predicted_area - intersection_area
        penalty = non_overlap_area / predicted_area
        return penalty

    def is_done(self, predicted_polygon, ground_truth_polygon):
        """
        We want the episode to end when:
        - the initial polygon covers 95% of the ground truth polygon
        - only 5% of the initial polygons area doesn't overlap with the ground truth
        """
        intersection_area = predicted_polygon.intersection(ground_truth_polygon).area
        predicted_area = predicted_polygon.area
        ground_truth_area = ground_truth_polygon.area

        overlap_ratio = intersection_area / ground_truth_area
        non_overlap_ratio = (predicted_area - intersection_area) / predicted_area

        return overlap_ratio >= 0.95 and non_overlap_ratio <= 0.05
    
class CustomCNNFeatureExtractor(BaseFeaturesExtractor):

    """
    Pretrained ResNet model to extract features from images to be used by the agent.
    """

    def __init__(self, observation_space: spaces.Dict):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim=512)

        # Pre-trained ResNet model from torchvision
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the fully connected layer of ResNet
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # The output features dimension of ResNet18 is 512
        self._features_dim = 512

    def forward(self, observations):
        # Extract image tensor from the observations dictionary
        image_tensor = observations["image"]
        
        # Ensure the image tensor has the correct shape
        if image_tensor.shape[1] == 80:
            image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert shape to [batch_size, height, width, channels]
            image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert shape to [batch_size, 3, height, width]

        # Process the image tensor through the CNN
        cnn_output = self.cnn(image_tensor)
        
        # Flatten the CNN output to get a 1D tensor
        cnn_output = th.flatten(cnn_output, 1)
        
        return cnn_output
    
# Enable interactive mode
plt.ion()

def real_time_visualisation(image, predicted_polygon, ground_truth_polygon):
    """
    Visualizes the predicted polygon and ground truth polygon over the input image.
    Updates the same plot window instead of creating a new one each time.
    """
    plt.clf()  # Clear the previous plot

    plt.imshow(image)
    
    # Convert predicted polygon and ground truth polygon to matplotlib format
    pred_poly_coords = np.array(predicted_polygon.exterior.coords)
    gt_poly_coords = np.array(ground_truth_polygon.exterior.coords)
    
    # Create polygons for display
    pred_patch = pltPolygon(pred_poly_coords, closed=True, edgecolor='r', fill=False, linewidth=2, label="Predicted Polygon")
    gt_patch = pltPolygon(gt_poly_coords, closed=True, edgecolor='g', fill=False, linewidth=2, label="Ground Truth Polygon")
    
    # Add polygons to plot
    plt.gca().add_patch(pred_patch)
    plt.gca().add_patch(gt_patch)
    
    plt.legend(loc='upper right')
    plt.title("Predicted vs Ground Truth Polygon")
    
    plt.draw()  # Redraw the current figure
    plt.pause(0.001)  # Small pause to allow the plot to update

# Disable interactive mode after training (optional)
def disable_interactive_mode():
    plt.ioff()
    plt.show()
    
def main():
    directory_path = r"C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\only_ships_images\\images"
    json_path = r"C:\\Users\\shiri\\Desktop\\Dataverze\\Vision_experimentation\\RL_attempt_3\\output_coco_valid_updated.json"
    threshold = 0
    contrast_factor = 5.0
    gamma = 1.2

    model = models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1')
    model.eval()
    transform = T.Compose([T.ToTensor()])

    polygons_data = process_directory(directory_path, json_path, model, transform, threshold, contrast_factor, gamma)

    total_epochs = 1 # change to greater number of epoch when greater amount of resources are available
    total_timesteps = 10000
    checkpoint_interval = 1000
    reward_log = []

    policy_kwargs = dict(
                features_extractor_class=CustomCNNFeatureExtractor,
                features_extractor_kwargs={}
            )

    # create the environment for the first image to initialise the model
    first_data = polygons_data[0]
    image_filename = first_data['image']
    initial_polygon = first_data['prediction']
    ground_truth_polygon = first_data['ground_truth']
    image = Image.open(os.path.join(directory_path, image_filename)).convert("RGB")
    env = (PolygonEnv(initial_polygon,ground_truth_polygon,image))

    # initialise the PPO model once
    model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs, device='cuda')
    
    # iterate over all images
    for epoch in range(total_epochs):
        print(f"Starting epoch {epoch + 1}/{total_epochs}")
        for image_index, data in enumerate(polygons_data):
            image_filename = data['image']
            initial_polygon = data['prediction']
            ground_truth_polygon = data['ground_truth']
            image = Image.open(os.path.join(directory_path, image_filename)).convert("RGB")
            
            # re-create environment for each new image
            # attempt to implement parallel environments at a later date
            env = (PolygonEnv(initial_polygon, ground_truth_polygon, image))
            model.set_env(env) # reuse existing model but set it to new environment

            for timestep in range(0, total_timesteps, checkpoint_interval):
                model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
                model.save(f"ppo_model_epoch_{epoch}_image_{image_index}_timestep_{timestep}")
                
                # Logging rewards
                rewards = []
                for i in range(100):
                    obs, _ = env.reset()
                    done = False
                    total_reward = 0
                    while not done:
                        action, _ = model.predict(obs)
                        obs, reward, done, _,_ = env.step(action)
                        # real time visualisation of moving vertices, comment out the line below to stop real-time visualisation
                        real_time_visualisation(image,env.current_polygon,env.ground_truth_polygon)
                        total_reward += reward
                    rewards.append(total_reward)
                reward_log.append({
                    'epoch': epoch,
                    'image_index': image_index,
                    'timestep': timestep,
                    'average_reward': np.mean(rewards)
                })
                
                # Save the reward log
                with open("reward_log.json", "w") as f:
                    json.dump(reward_log, f)
        
            print(f"Finished processing image {image_index + 1}/{len(polygons_data)} for epoch {epoch + 1}")
        print(f"Finished epoch {epoch + 1}/{total_epochs}")

    print("Training completed.")
    disable_interactive_mode()  # Keep the final plot visible

if __name__ == "__main__":
    main()

