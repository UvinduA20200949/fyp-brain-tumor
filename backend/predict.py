import tensorflow as tf
import pickle

from dst import dst_for_single

def ann_load_model(model_path='/Users/uvinduabeysinghe/Desktop/fyp/project-tumor-detection/backend/models/ann.keras'):
    ann_model = tf.keras.models.load_model(model_path)
    return ann_model

def ann_predictions(image,model):
    ann_prediction = model.predict(image)
    return ann_prediction

def svm_load_model(model_path='/Users/uvinduabeysinghe/Desktop/fyp/project-tumor-detection/backend/models/svmpickle_file'):
    svm_model = pickle.load(open(model_path, 'rb'))
    return svm_model

def svm_predictions(image, model):
    # Assuming `image` is a single image with shape (height, width, channels)
    # Flatten the image to shape (1, height*width*channels) for a single sample
    if image.ndim == 3:
        image = image.reshape(1, -1)
    # If `image` is already a batch of images, flatten each image
    elif image.ndim == 4:
        n_samples, height, width, channels = image.shape
        image = image.reshape(n_samples, height*width*channels)
    
    svm_predictions = model.predict_proba(image/255.0)
    
    return svm_predictions

def knn_load_model(model_path='/Users/uvinduabeysinghe/Desktop/fyp/project-tumor-detection/backend/models/knnpickle_file'):
    knn_model = pickle.load(open(model_path, 'rb'))
    return knn_model

def knn_predictions(image, model):
    if image.ndim == 3:
        image = image.reshape(1, -1)
    # If `image` is already a batch of images, flatten each image
    elif image.ndim == 4:
        n_samples, height, width, channels = image.shape
        image = image.reshape(n_samples, height*width*channels)
    
    knn_predictions = model.predict_proba(image/255.0)
    
    return knn_predictions

def reshape_array(arr, method='flatten'):
    """
    Reshape a numpy array from (1, 4) to (4,).

    Parameters:
    - arr: Input array of shape (1, 4).
    - method: Method to use for reshaping ('flatten' or 'reshape').

    Returns:
    - A numpy array of shape (4,).
    """
    if method == 'flatten':
        # Flatten the array to 1D
        return arr.flatten()
    elif method == 'reshape':
        # Explicitly reshape the array to 1D
        return arr.reshape(-1)
    else:
        raise ValueError("Method must be 'flatten' or 'reshape'.")
    
def reshape_predictions(ann_prediction, svm_prediction, knn_prediction):
    ann_prediction_new = reshape_array(ann_prediction)
    svm_prediction_new = reshape_array(svm_prediction) 
    knn_prediction_new = reshape_array(knn_prediction)

    return ann_prediction_new, svm_prediction_new, knn_prediction_new

def get_tumor_name(prediction):
    tumors = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    return tumors[prediction]

def make_predictions(image):
    ann_model = ann_load_model()
    ann_prediction = ann_predictions(image,ann_model)

    svm_model =  svm_load_model()
    svm_prediction = svm_predictions(image, svm_model)

    knn_model = knn_load_model()
    knn_prediction = knn_predictions(image, knn_model)

    print('ANN PREDICTIONS:',ann_prediction)
    print('KNN PREDICTIONS:',knn_prediction)
    print('SVM PREDICTIONS:',svm_prediction)

    ann_prediction_reshaped, svm_predictions_reshaped, knn_predictions_reshaped = reshape_predictions(ann_prediction, svm_prediction, knn_prediction)

    prediction_index = dst_for_single(ann_preds= ann_prediction_reshaped, svm_preds= svm_predictions_reshaped, knn_preds=knn_predictions_reshaped)
    prediction = get_tumor_name(prediction_index)

    print(prediction)

    return prediction
