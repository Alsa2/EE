from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model('maze_solver_model')

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
