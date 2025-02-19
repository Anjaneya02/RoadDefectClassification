from constants import *
from helper import *
from ConfusionMatrixHelper import confusion_matrix_and_classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

Mname='resnet50'
# Define the image size
IMAGE_SIZE = [224, 224, 3]

target_image_size = tuple(IMAGE_SIZE[:2])
# Load the model
resnet50 = ResNet50(include_top=False ,input_shape=IMAGE_SIZE, weights="imagenet")

# Visualize the model summary
# resnet50.summary()

for layer in resnet50.layers:
    layer.trainable = False

x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1536, activation='relu')(x)
prediction = Dense(6, activation='softmax')(x)


# Join it with the model
model = Model(inputs=resnet50.input, outputs=prediction)

model.summary()

# Define the learning rate
learning_rate = 0.0001

# Create an instance of the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the custom optimizer
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


train_path = constants.path_of_train_data
test_path = constants.path_of_test_data

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Train data
train_set = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# Test data
test_set = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

csv_logger = CSVLogger(
    filename="D:/ResearchPaperCode/hyperparameter/resnet5.csv", append=False
)


checkpoint = ModelCheckpoint(
    filepath="D:/ResearchPaperCode/hyperparameter/resnet5.keras", verbose=2, save_best_only=True
)
callbacks = [checkpoint, csv_logger]
start = datetime.now()
model_history = model.fit(
    train_set,
    validation_data=test_set,
    epochs=10,
    callbacks=callbacks,
)

duration = datetime.now() - start

print("Total elapsed time : ", duration)

val_data_path=test_path

class_names = sorted(os.listdir(val_data_path))
# Load validation data
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Rescale pixel values to [0, 1]
val_generator = datagen.flow_from_directory(
    val_data_path,
    target_size=target_image_size,  # Resize images to model input size
    batch_size=64,
    class_mode="categorical",
    shuffle=False  # Ensure the order matches for predictions
)
# Get true labels and corresponding class indices
y_true = val_generator.classes  # True labels
class_indices = val_generator.class_indices  # Class label mapping
class_names = list(class_indices.keys())  # Class names
# Predict labels
y_pred_prob = model.predict(val_generator)  # Predict probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert to class indices
# Classification report
print("Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
# Generate the classification report as a dictionary
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
# Convert the dictionary to a DataFrame
report_df = pd.DataFrame(report_dict).transpose()
# Save as a CSV file
report_df.to_csv("D:/ResearchPaperCode/hyperparameter/resnet_classification5.csv", index=True)
# Confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join('D:/ResearchPaperCode/hyperparameter', 'confusion_matrix_resnet5.png'))  # Save the confusion matrix
plt.close()