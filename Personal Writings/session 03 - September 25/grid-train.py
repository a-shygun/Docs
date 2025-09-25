import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight

df = pd.read_csv("./messidor_data.csv")  
df['file_path'] = df['id_code'].apply(lambda x: 'images/' + x)
paths = df['file_path'].to_list()
labels = pd.get_dummies(df['adjudicated_dme'], dtype=float).to_numpy()
NUM_CLASSES = labels.shape[1]

# USE_VAL_list = [False, True]
# BATCH_SIZE_list = [5, 10, 15, 20]
# IMG_SIZE_list = [(512, 512), (256, 256), (128, 128)]
# USE_AUG_list = [False, True]
# EPOCHS_list = [5, 10, 15]
USE_VAL_list = [True]
BATCH_SIZE_list = [5]
IMG_SIZE_list = [(128, 128), (256, 256), (512, 512)]
USE_AUG_list = [True]
EPOCHS_list = [10, 15, 20]
results_file = 'results.json'
try:
    with open(results_file, "r") as f:
        all_results = json.load(f)
except:
    all_results = []

# Compute total number of runs
total_runs = len(USE_VAL_list) * len(BATCH_SIZE_list) * len(IMG_SIZE_list) * len(USE_AUG_list) * len(EPOCHS_list)
run_count = 0

for USE_VAL in USE_VAL_list:
    for BATCH_SIZE in BATCH_SIZE_list:
        for IMG_SIZE in IMG_SIZE_list:
            for USE_AUG in USE_AUG_list:
                for EPOCHS in EPOCHS_list:
                    run_count += 1
                    rotation_range = 40 if USE_AUG else 0
                    width_shift_range = 0.2 if USE_AUG else 0.0
                    height_shift_range = 0.2 if USE_AUG else 0.0
                    shear_range = 0.2 if USE_AUG else 0.0
                    zoom_range = 0.2 if USE_AUG else 0.0
                    horizontal_flip = True if USE_AUG else False
                    vertical_flip = True if USE_AUG else False
                    brightness_range=[0.8,1.2]
                    channel_shift_range=20
                    fill_mode = 'nearest'

                    x_train, x_test, y_train, y_test = train_test_split(
                        paths, labels, test_size=0.1, random_state=0, stratify=labels
                    )
                    if USE_VAL:
                        x_train, x_val, y_train, y_val = train_test_split(
                            x_train, y_train, test_size=0.1, random_state=0, stratify=y_train
                        )

                    train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=rotation_range,
                        width_shift_range=width_shift_range,
                        height_shift_range=height_shift_range,
                        shear_range=shear_range,
                        zoom_range=zoom_range,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        fill_mode=fill_mode
                    )
                    val_datagen = ImageDataGenerator(rescale=1./255)
                    test_datagen = ImageDataGenerator(rescale=1./255)

                    def create_generator(file_paths, labels, datagen):
                        images = []
                        for f in file_paths:
                            img = tf.keras.utils.load_img(f, target_size=IMG_SIZE)
                            img_array = tf.keras.utils.img_to_array(img)
                            images.append(img_array)
                        images = np.array(images)
                        labels_arr = np.array(labels)
                        return datagen.flow(images, labels_arr, batch_size=BATCH_SIZE, shuffle=True)

                    train_gen = create_generator(x_train, y_train, train_datagen)
                    test_gen = create_generator(x_test, y_test, test_datagen)
                    steps_train = len(x_train) // BATCH_SIZE
                    steps_test = len(x_test) // BATCH_SIZE

                    if USE_VAL:
                        val_gen = create_generator(x_val, y_val, val_datagen)
                        steps_val = len(x_val) // BATCH_SIZE

                    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    x = Dense(256, activation='relu')(x)
                    # x = Dropout(0.5)(x)
                    # x = Dense(128, activation='relu')(x)
                    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
                    model = Model(inputs=base_model.input, outputs=predictions)

                    for layer in base_model.layers:
                        layer.trainable = False
                    # for layer in base_model.layers[-10:]:
                    #     layer.trainable = True

                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    # model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    
                    y_train_labels = np.argmax(y_train, axis=1)
                    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
                    class_weights_dict = dict(enumerate(class_weights))

                    # model.fit(train_gen, validation_data=val_gen if USE_VAL else None, epochs=EPOCHS, class_weight=class_weights_dict)
                    model.fit(
                        train_gen,
                        validation_data=val_gen if USE_VAL else None,
                        epochs=EPOCHS
                    )

                    loss, acc = model.evaluate(test_gen, steps=steps_test)

                    entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "run_number": run_count,
                        "total_runs": total_runs,
                        "accuracy": f"{acc:.4f}",
                        "batch_size": BATCH_SIZE,
                        "img_size": IMG_SIZE,
                        "epochs": EPOCHS,
                        "num_classes": NUM_CLASSES,
                        "use_val": USE_VAL,
                        "use_aug": USE_AUG,
                        "augmentation": {
                            "rotation_range": rotation_range,
                            "width_shift_range": width_shift_range,
                            "height_shift_range": height_shift_range,
                            "shear_range": shear_range,
                            "zoom_range": zoom_range,
                            "horizontal_flip": horizontal_flip,
                            "vertical_flip": vertical_flip,
                            "rescale": 1./255
                        }
                    }

                    all_results.append(entry)

                    # Save JSON and CSV after each run
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=4)

                    df_results = pd.json_normalize(all_results, sep='_')
                    df_results.to_csv('results.csv', index=False)

                    # Print progress
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Run {run_count}/{total_runs} finished. Accuracy: {acc:.4f}")