from tensorflow.keras.models import load_model

model_path = "maze_solver_model"  # Path to the directory containing the model files
model = load_model(model_path)

from tensorflow.keras.utils import plot_model

# Specify the path where you want to save the model visualization
# For example, "./maze_solver_model_visualization.png"
output_path = "./maze_solver_model_visualization.png"

# Plot the model and save it to the specified path
plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
