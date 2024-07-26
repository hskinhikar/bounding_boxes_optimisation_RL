import gymnasium as gym
from gymnasium import spaces
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
import json
import collections
from gen_predicted_grndreality_polygons import process_directory

class PolygonEnv(gym.Env):

    """
    Custom gym environment that sets up the reinforcement learning environmetn such that:
    - it observes the vertices of the initially predicted polygon as well as the image itself
    - it moves the vertices of the initial polygon to maximise rewards from the reward policy
    Reward policy:
    - positive reward for IoU overlap between the predicted polygon and ground truth polygon
    ^ this encourages the initial polygon to cover as much of ground truth polygon as possible
    - negative reward for parts of predicted polygon that don't overlap with the ground truth polygon
    ^ this is to avoid the initial polygon from covering the entire image as opposed to just the ground truth
    - negative reward for actions that make the edges of the polygon intersect themselves
    ^ avoid the initial polygon from being an invalid shape of which IoU cannot be taken
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

        obs_array = self._dict_to_array(observations)
        info = {}
        print(f"PolygonEnv.reset: {type(obs_array)}")
        return obs_array, info

    def step(self, action):
        """
        Takes an action and returns next observation as a dictionary, along with reward, done, truncated and info
        Converts observation into a numpy array using _dict_to_array
        returns numpy array, reward, done, truncated and info
        """
        vertices = np.array(self.current_polygon.exterior.coords)[:-1]# Current vertices of polygon
        adjusted_vertices = vertices + action.reshape(-1, 2)# Adjust vertices based on action
        new_polygon = Polygon(adjusted_vertices)# Create new polygon with adjusted vertices
        new_polygon = orient(new_polygon, sign=1.0)

        if not new_polygon.is_valid:
            reward = -1.0#penalise if new polygon is not valid
        else:
            self.current_polygon = new_polygon
            iou = self.calculate_iou(self.current_polygon, self.ground_truth_polygon)
            non_overlap_penalty = self.calculate_non_overlap_penalty(self.current_polygon, self.ground_truth_polygon)
            reward = iou - (1 - iou) - non_overlap_penalty # Compite reward

        done = self.is_done(self.current_polygon, self.ground_truth_polygon)
        info = {}
        truncated = False

        obs = {
            "image": np.array(self.image).transpose(2, 0, 1),  # convert image to numpy array with shape (channels, height, width)
            "vertices": np.array(self.current_polygon.exterior.coords)[:-1].flatten() # flatten polygons vertices to 1D
        }

        obs_array = self._dict_to_array(obs)
        return obs_array, reward, done, truncated, info

    def _dict_to_array(self, obs_dict):
        # flatten and combine image and vertices of initial polygon into a 1 dimensional array
        image_flat = obs_dict["image"].flatten()
        vertices_flat = obs_dict["vertices"].flatten()
        return np.concatenate([image_flat, vertices_flat])


    
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


class GymToStableBaselines3Wrapper(gym.Wrapper):

    """
    Wrapper for converting observations from a dictionary to a numpy array which is needed by Stable Baselines 3 (the RL library used).
    Designed to wrap around a single environment.
    """

    def __init__(self, env):

        super(GymToStableBaselines3Wrapper, self).__init__(env)

    def step(self, action):
        """
        Calls PolygonEnv.step(action)
        Converts the dictionary observation to a numpy array if its not already
        Returns the numpy array, reward, done and info
        """
        obs, reward, done, truncated, info = self.env.step(action)
        done = done or truncated
        obs = self._convert_obs(obs)
        print(f"GymToStableBaselines3Wrapper.step: {type(obs)}")
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls PolygonEnv.reset()
        Converts dictionary observation to a numpy array if it is not already
        returns numpy array and info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        print(f"GymToStableBaselines3Wrapper reset: obs type: {type(obs)}, info type: {type(info)}")
        if isinstance(obs, tuple):
            obs = obs[0]  # Ensures that the observations are not in the form of a tuple
        obs = self._convert_obs(obs)
        return obs, info

    def _convert_obs(self, obs):
        """
        Converts observation dictionaries into numpy arrays 
        """
        if isinstance(obs, (dict, collections.OrderedDict)):
            image_flat = obs["image"].flatten()
            vertices_flat = obs["vertices"].flatten()
            return np.concatenate([image_flat, vertices_flat])
        return obs

    def _dict_to_array(self, obs_dict): # to flatten observations into 1 dimensional arrays
        image_flat = obs_dict["image"].flatten()
        vertices_flat = obs_dict["vertices"].flatten()
        return np.concatenate([image_flat, vertices_flat])


class CustomDummyVecEnv(DummyVecEnv):
    """
    Manages multiple environments, ensuring efficient parallel proessing.
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)

    def reset(self):
        """
        Calls GymToStableBaselines3Wrapper.reset()
        Aggregates(collects) the observations from all environments
        Converts the aggregated observation to a numpy array
        returns the aggregated numpy array and info dictionaries
        """
        obs = super().reset()
        print(f"CustomDummyVecEnv reset: {type(obs)}")
        obs_array = self._convert_obs(obs)
        print(f"Converted CustomDummyVecEnv reset: {type(obs_array)}")
        self._save_obs(0, obs_array)
        infos = [{} for _ in range(self.num_envs)]
        return self.buf_obs, infos

    def step_wait(self):
        """
        Calls GymToStableBaselines3Wrapper.step(action) for each environment
        Aggregated (collects) observations, rewards, dones and infos from all environments
        Converts aggregated observation to a numpy array
        returns aggregated numpy array, rewards, dones and infos
        """
        results = [env.step(action) for env, action in zip(self.envs, self.actions)]
        obs, rews, dones, infos = zip(*results)
        print(f"CustomDummyVecEnv step_wait: obs types: {[type(o) for o in obs]}")
        obs_array = [self._convert_obs(o) for o in obs]
        print(f"Converted CustomDummyVecEnv step_wait: {type(obs_array)}")
        self._save_obs(0, obs_array)
        return self.buf_obs, np.array(rews), np.array(dones), infos

    def _save_obs(self, env_idx, obs):
        if isinstance(obs, np.ndarray):
            self.buf_obs[env_idx] = obs
        else:
            raise TypeError(f"Observations must be numpy arrays, got {type(obs)} instead.")

    def _convert_obs(self, obs):
        if isinstance(obs, (dict, collections.OrderedDict)):
            image_flat = obs["image"].flatten()
            vertices_flat = obs["vertices"].flatten()
            return np.concatenate([image_flat, vertices_flat])
        return obs


class CustomVecTransposeImage(VecEnvWrapper):

    """
    Custom vectorized environment wrapper that processes the observations before passing them to RL 
    in a format that is suitable for processing by the agent.
    Extends GymToStavleBaselines3Wrapper so that it can handle multiple environments simultaneously.
    ^ essential for efficient RL training

    """

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        """
        Calls CustomDummyVecEnv.reset()
        Processes the aggregated observation to ensure it is a numpy array
        returns the processed numpy array and info dictionaries
        """
        observations, infos = self.venv.reset()
        print(f"CustomVecTransposeImage reset: {type(observations)}")
        processed_obs = self.process(observations)
        print(f"Processed CustomVecTransposeImage reset: {type(processed_obs)}")
        return processed_obs, infos

    def step(self, actions):
        """
        Calls CustomDummyVecEnv.step_wait()
        Processes aggregated observation to ensure it is a numpy array
        returns processes numpy array, rewards, dones and infos
        """
        observations, rewards, dones, infos = self.venv.step(actions)
        print(f"CustomVecTransposeImage step: {type(observations)}")
        processed_obs = self.process(observations)
        print(f"Processed CustomVecTransposeImage step: {type(processed_obs)}")
        return processed_obs, rewards, dones, infos

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        print(f"CustomVecTransposeImage step_wait: {type(observations)}")
        processed_obs = self.process(observations)
        print(f"Processed CustomVecTransposeImage step_wait: {type(processed_obs)}")
        return processed_obs, rewards, dones, infos

    def process(self, observations):
        if isinstance(observations, list):
            processed_obs = np.array([self._convert_obs(obs) for obs in observations])
        elif isinstance(observations, np.ndarray):
            processed_obs = observations
        elif isinstance(observations, (dict, collections.OrderedDict)):
            processed_obs = np.array([self._convert_obs(observations)])
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")

        print(f"Process method output: {type(processed_obs)}")
        return processed_obs

    def _convert_obs(self, obs):
        if isinstance(obs, (dict, collections.OrderedDict)):
            image_flat = obs["image"].flatten()
            vertices_flat = obs["vertices"].flatten()
            return np.concatenate([image_flat, vertices_flat])
        return obs

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

    total_epochs = 3
    total_timesteps = 10000
    checkpoint_interval = 1000
    reward_log = []

    for epoch in range(total_epochs):
        print(f"Starting epoch {epoch + 1}/{total_epochs}")
        for image_index, data in enumerate(polygons_data):
            image_filename = data['image']
            initial_polygon = data['prediction']
            ground_truth_polygon = data['ground_truth']
            image = Image.open(os.path.join(directory_path, image_filename)).convert("RGB")
            
            #create environment and apply wrappers
            env = CustomDummyVecEnv([lambda: GymToStableBaselines3Wrapper(PolygonEnv(initial_polygon, ground_truth_polygon, image))])
            env = CustomVecTransposeImage(env)
            
            policy_kwargs = dict(
                features_extractor_class=CustomCNNFeatureExtractor,
                features_extractor_kwargs={}
            )

            model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
            for timestep in range(0, total_timesteps, checkpoint_interval):
                print(f"Before learning step: env reset obs type: {type(env.reset()[0])}")
                model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
                model.save(f"ppo_model_epoch_{epoch}_image_{image_index}_timestep_{timestep}")
                
                # Logging rewards
                rewards = []
                for i in range(100):
                    obs, _ = env.reset()
                    print(f"During reward logging: obs type: {type(obs)}")
                    done = False
                    total_reward = 0
                    while not done:
                        action, _ = model.predict(obs)
                        obs, reward, done, info = env.step(action)
                        print(f"Step observation type: {type(obs)}")
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

if __name__ == "__main__":
    main()

"""
Output:
PolygonEnv.reset: <class 'numpy.ndarray'>
GymToStableBaselines3Wrapper reset: obs type: <class 'numpy.ndarray'>, info type: <class 'dict'>
CustomDummyVecEnv reset: <class 'collections.OrderedDict'>
Converted CustomDummyVecEnv reset: <class 'numpy.ndarray'>
CustomVecTransposeImage reset: <class 'collections.OrderedDict'>
Process method output: <class 'numpy.ndarray'>
Processed CustomVecTransposeImage reset: <class 'numpy.ndarray'>
Before learning step: env reset obs type: <class 'numpy.ndarray'>
PolygonEnv.reset: <class 'numpy.ndarray'>
GymToStableBaselines3Wrapper reset: obs type: <class 'numpy.ndarray'>, info type: <class 'dict'>       
CustomDummyVecEnv reset: <class 'collections.OrderedDict'>
Converted CustomDummyVecEnv reset: <class 'numpy.ndarray'>
CustomVecTransposeImage reset: <class 'collections.OrderedDict'>
Process method output: <class 'numpy.ndarray'>
Processed CustomVecTransposeImage reset: <class 'numpy.ndarray'>
...
Traceback (most recent call last):
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\environment_and_run.py", line 416, in <module>
    main()
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\environment_and_run.py", line 383, in main
    model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\myenv\lib\site-packages\stable_baselines3\ppo\ppo.py", line 315, in learn
    return super().learn(
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\myenv\lib\site-packages\stable_baselines3\common\on_policy_algorithm.py", line 300, in learn
    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\myenv\lib\site-packages\stable_baselines3\common\on_policy_algorithm.py", line 178, in collect_rollouts
    obs_tensor = obs_as_tensor(self._last_obs, self.device)
  File "c:\Users\shiri\Desktop\Dataverze\Vision_experimentation\RL_attempt_3\myenv\lib\site-packages\stable_baselines3\common\utils.py", line 489, in obs_as_tensor
    raise Exception(f"Unrecognized type of observation {type(obs)}")
...
Exception: Unrecognized type of observation <class 'tuple'>


At this point I'm not sure at which point data being input into the PPO model is being converted into a tuple.
Steps to fix this coule be to add debug statements within Stable Baselines3 library functions which are used to initialise and train the model.
Alternatively, we can create a custom wrapper around the 'learn' method of PPO to capture and print observations before it is passed into 'obs_as_tensor'
"""
