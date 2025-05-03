import os
import sys
import json
import numpy as np
import tensorflow as tf
import logging
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

tf_version = tf.__version__
print(f"Using TensorFlow version: {tf_version}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): 
            return None
        return super(NumpyEncoder, self).default(obj)

class Config:
    BASE_DIR = Path("food_dataset")
    PROCESSED_DIR = BASE_DIR / "processed"
    MODEL_DIR = Path("models")
    LOGS_DIR = Path("logs")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    INITIAL_LR = 1e-4
    MIN_LR = 1e-6
    USE_DATA_AUGMENTATION = True
    RANDOM_SEED = 42
    FINE_TUNE_AT = 0.7

try:
    from efficientnet_v2 import EfficientNetV2S
    print("EfficientNetV2 package already installed")
except ImportError:
    print("Installing EfficientNetV2 package...")
    os.system("pip install git+https://github.com/sebastian-sz/efficientnet-v2-keras@main")
    from efficientnet_v2 import EfficientNetV2S

class GPUManager:
    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                logger.info(f"GPU available: {len(gpus)} device(s)")
                return True
            except RuntimeError as e:
                logger.error(f"GPU configuration error: {e}")
        logger.warning("No GPU available, using CPU")
        return False

class DataManager:
    def __init__(self):
        if not Config.PROCESSED_DIR.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {Config.PROCESSED_DIR}. "
                "Please run the dataset downloader script first."
            )
        Config.MODEL_DIR.mkdir(exist_ok=True)
        Config.LOGS_DIR.mkdir(exist_ok=True)
        np.random.seed(Config.RANDOM_SEED)
        tf.random.set_seed(Config.RANDOM_SEED)
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        logger.info(f"Found {self.num_classes} classes")
    
    def _get_class_names(self):
        train_dir = Config.PROCESSED_DIR / "train"
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        class_names_path = Config.BASE_DIR / "data" / "class_names.json"
        if not class_names_path.exists():
            class_names_path.parent.mkdir(exist_ok=True)
            with open(class_names_path, 'w') as f:
                json.dump(class_names, f, cls=NumpyEncoder)
        return class_names
    
    def create_data_generators(self):
        valid_datagen = ImageDataGenerator(rescale=1./255)
        if Config.USE_DATA_AUGMENTATION:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            logger.info("Using data augmentation for training")
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            Config.PROCESSED_DIR / "train",
            target_size=Config.IMG_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=Config.RANDOM_SEED
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            Config.PROCESSED_DIR / "val",
            target_size=Config.IMG_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = valid_datagen.flow_from_directory(
            Config.PROCESSED_DIR / "test",
            target_size=Config.IMG_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator

class ModelBuilder:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build_model(self):
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(*Config.IMG_SIZE, 3)
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs, name="FoodClassifier_EfficientNetV2S")
        
        base_model.trainable = False
        
        self.base_model = base_model
        
        if Config.FINE_TUNE_AT > 0:
            self.fine_tune_at = int(len(base_model.layers) * Config.FINE_TUNE_AT)
        else:
            self.fine_tune_at = 0
        
        model.compile(
            optimizer=Adam(learning_rate=Config.INITIAL_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Model built: EfficientNetV2S")
        logger.info(f"Total layers: {len(model.layers)}")
        logger.info(f"Trainable layers (initial): {len([l for l in model.layers if l.trainable])}")
        
        return model
    
    def unfreeze_for_fine_tuning(self, model):
        self.base_model.trainable = True
        
        model.compile(
            optimizer=Adam(learning_rate=Config.INITIAL_LR / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Fine-tuning: Base model unfrozen")
        logger.info(f"Trainable layers (fine-tuning): {len([l for l in model.layers if l.trainable])}")
        
        return model

class ModelTrainer:
    def __init__(self, model, data_manager):
        self.model = model
        self.data_manager = data_manager
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.model_save_dir = Config.MODEL_DIR / f"{self.timestamp}_efficientnetv2"
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Config.LOGS_DIR / f"{self.timestamp}_efficientnetv2"
        self.logs_dir.mkdir(exist_ok=True)
        
        self.history_dict = {}
    
    def create_callbacks(self):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            ModelCheckpoint(
                filepath=self.model_save_dir / "best_model.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=Config.MIN_LR,
                verbose=1
            ),
            
            TensorBoard(
                log_dir=self.logs_dir,
                histogram_freq=1,
                update_freq='epoch'
            ),
            
            CSVLogger(
                filename=self.logs_dir / "training_log.csv",
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train(self, builder):
        train_gen, val_gen, test_gen = self.data_manager.create_data_generators()
        
        with open(self.logs_dir / "model_summary.txt", 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        callbacks = self.create_callbacks()
        
        logger.info("Phase 1: Training classification head")
        history1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=int(Config.EPOCHS * 0.3),
            callbacks=callbacks,
            verbose=1
        )
        
        for key, values in history1.history.items():
            self.history_dict[key] = [float(val) for val in values]
        
        logger.info("Phase 2: Fine-tuning")
        self.model = builder.unfreeze_for_fine_tuning(self.model)
        
        history2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.EPOCHS,
            initial_epoch=len(history1.history['loss']),
            callbacks=callbacks,
            verbose=1
        )
        
        for key, values in history2.history.items():
            if key in self.history_dict:
                self.history_dict[key].extend([float(val) for val in values])
            else:
                self.history_dict[key] = [float(val) for val in values]
        
        final_model_path = self.model_save_dir / "final_model.h5"
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        saved_model_path = self.model_save_dir / "saved_model"
        tf.keras.models.save_model(self.model, saved_model_path)
        logger.info(f"SavedModel format saved to {saved_model_path}")
        
        self._evaluate_model(test_gen)
        
        self._plot_training_history()
        
        return self.history_dict, final_model_path
    
    def _evaluate_model(self, test_gen):
        try:
            test_gen.reset()
            
            logger.info("Evaluating model on test set")
            results = self.model.evaluate(test_gen, verbose=1)
            
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = float(results[i])
            
            with open(self.logs_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, cls=NumpyEncoder, indent=2)
            
            logger.info(f"Test set metrics: {metrics}")
            
            test_gen.reset()
            y_pred = self.model.predict(test_gen)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            y_true = test_gen.classes
            
            class_names = list(test_gen.class_indices.keys())
            report = classification_report(y_true, y_pred_classes, target_names=class_names)
            
            with open(self.logs_dir / "classification_report.txt", 'w') as f:
                f.write(report)
            
            cm = confusion_matrix(y_true, y_pred_classes)
            
            plt.figure(figsize=(15, 15))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(self.logs_dir / "confusion_matrix.png")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
    
    def _plot_training_history(self):
        try:
            with open(self.logs_dir / "history.json", 'w') as f:
                json.dump(self.history_dict, f, cls=NumpyEncoder, indent=2)
            
            pd.DataFrame(self.history_dict).to_csv(self.logs_dir / "history.csv", index=False)
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history_dict['loss'], label='Training Loss')
            plt.plot(self.history_dict['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history_dict['accuracy'], label='Training Accuracy')
            plt.plot(self.history_dict['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            
            plt.tight_layout()
            plt.savefig(self.logs_dir / "training_history.png")
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description='Train food classification model with EfficientNetV2')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation',
                        help='Disable data augmentation')
    
    args = parser.parse_args()
    
    Config.BATCH_SIZE = args.batch_size
    Config.EPOCHS = args.epochs
    Config.USE_DATA_AUGMENTATION = args.augmentation
    
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    GPUManager.setup_gpu()
    
    logger.info(f"Starting food classifier training with EfficientNetV2S")
    logger.info(f"Training config: batch_size={Config.BATCH_SIZE}, epochs={Config.EPOCHS}, "
                f"augmentation={Config.USE_DATA_AUGMENTATION}")
    
    try:
        data_manager = DataManager()
        
        builder = ModelBuilder(data_manager.num_classes)
        model = builder.build_model()
        
        trainer = ModelTrainer(model, data_manager)
        history, model_path = trainer.train(builder)
        
        logger.info(f"Training completed successfully. Model saved to {model_path}")
        
        class_mapping = {str(i): name for i, name in enumerate(data_manager.class_names)}
        with open(Path(model_path).parent / "class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
