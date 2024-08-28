### Bounding Boxes Optimization with Reinforcement Learning

This project utilises reinforcement learning to optimize bounding boxes generated around ships in satellite imagery, leveraging Mask RCNN models. Below is an overview of the repository's contents:




- **drawing_bounding_boxes.py**: Draws bounding boxes around ships using Mask RCNN.
- **display_multiple_polygon_predictions_as_is.py**: Displays predicted polygons alongside ground truth and processed input images.
- **multiple_verticed_polygons_displayed.py**: Modifies predictions to limit polygons to eight vertices.
- **both_polygons_displayed.py**: Ensures only the highest-confidence polygon is displayed alongside the ground truth.
- **gen_predicted_grndreality_polygons.py**: Generates initial predicted and ground reality polygons for reinforcement learning.
- **reinforcement_learning_script.py**: The main script for reinforcement learning, currently under development.
- **agent_actively_moving_vertices.mp4**: Video showing the agent actively moving the vertices of the polygon to maximise rewards 

Explanation documents:
- **Bounding_analysis_working_doc.pdf**: A detailed document outlining the thought process, diversions, and progress of the project.
- **Reinforcement_learning_breakdown.pdf**: Will be updated once the working reinforcement script is finalised

This project is actively under development, with ongoing work to improve the reinforcement learning aspects. The working documents provide in-depth explanations of the methodologies and progress. I would recommend reading through the 'task' section of the Reinforcement_learning_breakdown.pdf for an overview of the task
