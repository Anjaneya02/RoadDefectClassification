from helper import *
import constants
from constants import *
from ConfusionMatrixHelper import confusion_matrix_and_classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

Mname=MobileNetV2
# Define the image size
IMAGE_SIZE = [224, 224, 3]
target_image_size = tuple(IMAGE_SIZE[:2])

# Load the model
mobilenetv2 = MobileNetV2(include_top=False, input_shape=IMAGE_SIZE, weights='imagenet')

# Visualize the model summary
#mobilenetv2.summary()

for layer in mobilenetv2.layers:
    layer.trainable = False


# Add custom head to the base model

x = mobilenetv2.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=mobilenetv2.input, outputs=predictions)


adam=Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.summary()

train_path = constants.path_of_train_data
test_path = constants.path_of_test_data



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Train data
train_set = train_datagen.flow_from_directory(train_path,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')

# Test data
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

# Define the file name for the model checkpoint
checkpoint_filepath = "D:/ResearchPaperCode/models/mobilenet.keras"

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath, verbose=1, save_best_only=True
)

csv_logger = CSVLogger(filename="D:/ResearchPaperCode/csv/mobilenet.csv", append=True)


# Combine all callbacks
callbacks = [checkpoint, csv_logger]

# Start timing
start = datetime.now()

# Train the model
model_history = model.fit(train_set,
                          validation_data=test_set,
                          epochs=constants.no_of_epoch,
                          callbacks=callbacks)
# Calculate duration5
duration = datetime.now() - start

print('Total elapsed time:', duration)

val_data_path=test_path

class_names = sorted(os.listdir(val_data_path))
# Load validation data
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Rescale pixel values to [0, 1]
val_generator = datagen.flow_from_directory(
    val_data_path,
    target_size=target_image_size,  # Resize images to model input size
    batch_size=32,
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
report_df.to_csv("D:/ResearchPaperCode/ModelCodeFiles/classification/mobilenet.csv", index=True)
# Confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join('D:/ResearchPaperCode/ModelCodeFiles/confusion', 'confusion_matrix_mobilenet.png'))  # Save the confusion matrix
plt.close()