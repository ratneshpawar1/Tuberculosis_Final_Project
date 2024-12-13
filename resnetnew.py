import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def load_and_preprocess_images(folder, target_size):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img / 255.0
            images.append(img)
    if not images:
        raise ValueError(f"No images found in the folder: {folder}")
    return images

target_size = (224, 224)

normal_images = load_and_preprocess_images('normal', target_size)
tuberculosis_images = load_and_preprocess_images('Tuberculosis', target_size)

all_images = normal_images + tuberculosis_images
labels = [0] * len(normal_images) + [1] * len(tuberculosis_images)

all_images = np.array(all_images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    all_images, labels, test_size=0.3, random_state=42, stratify=labels
)

num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

y_train_labels = np.argmax(y_train, axis=1)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))

def build_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

input_shape = (target_size[0], target_size[1], 3)
model = build_model(input_shape, num_classes)
optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_format='h5'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

callbacks = [early_stopping, checkpoint, reduce_lr]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

report = classification_report(y_true_labels, y_pred_labels)
print(report)

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

auc = roc_auc_score(y_true_labels, y_pred[:, 1])
print('ROC AUC Score:', auc)
f1 = f1_score(y_true_labels, y_pred_labels)
print('F1 Score:', f1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

def get_gradcam(model, img_array, layer_name):
    grad_model = Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam.numpy(), (target_size[1], target_size[0]))
    return cam

def overlay_heatmap(img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    overlaid_img = heatmap * 0.4 + img
    overlaid_img = overlaid_img / np.max(overlaid_img)
    return overlaid_img

tuberculosis_indices = np.where(y_true_labels == 1)[0]
sample_indices = random.sample(list(tuberculosis_indices), min(5, len(tuberculosis_indices)))
plt.figure(figsize=(15, 7))
plt.suptitle('Tuberculosis Chest X-rays with Grad-CAM Heatmap Overlay', fontsize=16)
for idx, i in enumerate(sample_indices):
    img = X_test[i]
    img_array = np.expand_dims(img, axis=0)
    heatmap = get_gradcam(model, img_array, 'block5_conv3')
    overlaid_img = overlay_heatmap(img, heatmap)
    plt.subplot(1, 5, idx + 1)
    plt.imshow(overlaid_img)
    plt.axis('off')
plt.tight_layout()
plt.show()

misclassified_indices = np.where(y_pred_labels != y_true_labels)[0]

plt.figure(figsize=(15, 7))
plt.suptitle('Misclassified Images', fontsize=16)
for idx, i in enumerate(misclassified_indices[:5]):
    img = X_test[i]
    true_label = y_true_labels[i]
    pred_label = y_pred_labels[i]
    plt.subplot(1, 5, idx + 1)
    plt.imshow(img)
    plt.title(f'True: {true_label}, Pred: {pred_label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
acc_per_fold = []
loss_per_fold = []
for train_index, val_index in kf.split(X_train):
    print(f'Training for fold {fold_no} ...')
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    y_fold_train_labels = np.argmax(y_fold_train, axis=1)
    class_weights_fold = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_fold_train_labels),
        y=y_fold_train_labels
    )
    class_weights_fold = dict(enumerate(class_weights_fold))
    model_fold = build_model(input_shape, num_classes)
    model_fold.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history_fold = model_fold.fit(
        datagen.flow(X_fold_train, y_fold_train, batch_size=32),
        epochs=20,
        validation_data=(X_fold_val, y_fold_val),
        class_weight=class_weights_fold,
        callbacks=[early_stopping, reduce_lr]
    )
    scores = model_fold.evaluate(X_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model_fold.metrics_names[0]} of {scores[0]}; {model_fold.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_no += 1

print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold):.2f}% (+- {np.std(acc_per_fold):.2f}%)')
print(f'> Loss: {np.mean(loss_per_fold):.4f}')
