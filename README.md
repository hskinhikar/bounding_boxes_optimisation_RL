### Bounding Boxes Optimization with Reinforcement Learning

This project utilizes reinforcement learning to optimize bounding boxes generated around ships in satellite imagery, leveraging Mask RCNN models. Below is an overview of the repository's contents:

- **Bounding_analysis_working_doc.pdf**: A detailed document outlining the thought process, diversions, and progress of the project.
- **Reinforcement_learning_breakdown.pdf**: Explains reinforcement learning concepts and their application within the project.
- **environment_and_run.py**: The main script for reinforcement learning, currently under development.
- **gen_predicted_grndreality_polygons.py**: Generates initial predicted and ground reality polygons for reinforcement learning.
- **drawing_bounding_boxes.py**: Draws bounding boxes around ships using Mask RCNN.
- **display_multiple_polygon_predictions_as_is.py**: Displays predicted polygons alongside ground truth and processed input images.
- **multiple_verticed_polygons_displayed.py**: Modifies predictions to limit polygons to eight vertices.
- **both_polygons_displayed.py**: Ensures only the highest-confidence polygon is displayed alongside the ground truth.

This project is actively under development, with ongoing work to improve the reinforcement learning aspects. The working documents provide in-depth explanations of the methodologies and progress.
