This project uses reinforcement learning to optimise outlines generated around ships by Mask RCNN models.

I have uploaded a working document which explains my thought process, diversions and overall progress of this project. Please refer to it for more indepth information about goals of the project - I am currently working on documentation which explains reinforcement learning and how it was applied in this context. Note: the reinforcement learning script is not yet functional and I am still currently working on it.

The scripts corresponding to images shown in the working document are as follows:

Script that uses Mask RCNN model to draw outlines around ships in satellite imagery - drawing_bounding_boxes.py

Script that displayed the predicted polygon (using Mask RCNN model) alongside the ground reality as well as displaying the input imagery post processing - display_multiple_polygon_predictions_as_is.py.

Script that alters the predictions so that the predicted polygon has only 8 vertices (explanation for this provided in the working document) - multiple_verticed_polygon_displayed.py.

Script that ensures only one 8-vertices polygon is predicted by the Mask RCNN model (displays the polygon with the highest confidence rating) and displayed alongside ground reality polygons - both_polygons_displayed.py.

The following script is meant to be used in conjuction with the reinforcement learning script. This script generates intial_predicted_polygon and ground_reality_polygon which are then imported into the reinforcement learning script - gen_predicted_grndreality_polygons.py

The reinforcement learning script (this script is still in the process of being debugged and a separate document will explain how reinforcement learning as been applied as well as flow of code) - environment_and_run.py
