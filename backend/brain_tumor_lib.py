from preprocess import preprocess_image
from predict import make_predictions
from generate_descriptions import get_description
from database_operations import create_session, insert_data_to_table, get_all_brain_tumor_tests, get_patient_data

def brain_tumor_predict(image,patient_name):
    image_segmented_base64 , preprocessed_image = preprocess_image(image)
    prediction = make_predictions(preprocessed_image)
    tumor_description = get_description(prediction)

    session = create_session()
    insert_data_to_table(session, patient_name, image , prediction)

    return image_segmented_base64, prediction, tumor_description

def get_all_records():
    session = create_session()
    all_records = get_all_brain_tumor_tests(session)
    return all_records

def fetch_patient_history(patient_name):
    session = create_session()
    patient_records = get_patient_data(session, patient_name)
    return patient_records

