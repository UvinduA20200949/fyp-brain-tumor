import datetime
import base64

from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.reflection import Inspector

# Assuming Base is defined elsewhere
Base = declarative_base()

class BrainTumorTest(Base):
    __tablename__ = 'brain_tumor_test'
    
    id = Column(Integer, primary_key=True)
    patient_name = Column(String, nullable=False)
    image = Column(LargeBinary, nullable=False)
    predicted_tumor = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False)

def insert_brain_tumor_test_record(session, patient_name, image_base64, predicted_tumor, created_date):
    # Decode the base64 string to get the binary image data
    image_data = base64.b64decode(image_base64)
    
    # Create a new record with the decoded image data
    new_record = BrainTumorTest(patient_name=patient_name, image=image_data, predicted_tumor=predicted_tumor, created_date=created_date)
    
    try:
        session.add(new_record)
        session.commit()
        print("Record inserted successfully.")
    except Exception as e:
        session.rollback()
        print(f"Failed to insert record: {e}")

def create_session():
    DATABASE_URI = 'mysql+pymysql://root:MySQL1031@localhost/brain_tumor'

    engine = create_engine(DATABASE_URI, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
    
def insert_data_to_table(session, patient_name, image_path, tumor):
    insert_brain_tumor_test_record(session, patient_name, image_path, tumor, datetime.datetime.now())

def get_patient_data(session, patient_name):
    """
    Fetches records from the 'brain_tumor_test' table for a given patient name.

    :param session: SQLAlchemy Session object through which database queries will be executed.
    :param patient_name: The name of the patient for whom data is to be fetched.
    :return: A list of dictionaries where each dictionary represents the data of a record matching the patient name.
    """
    try:
        # Querying the database for all records that match the provided patient name.
        records = session.query(BrainTumorTest).filter(BrainTumorTest.patient_name == patient_name).all()
        
        # Preparing the list of dictionaries to return.
        results = []
        for record in records:
            results.append({
                "id": record.id,
                "patient_name": record.patient_name,
                "predicted_tumor": record.predicted_tumor,
                "created_date": record.created_date.strftime("%Y-%m-%d %H:%M:%S"),
                # If you intend to return the image data, consider converting it to a suitable format (e.g., base64)
                # "image": base64.b64encode(record.image).decode('utf-8')
            })
        
        return results

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return []
    
def get_all_brain_tumor_tests(session):    
    try:
        # Query the database for all entries in the brain_tumor_test table
        all_entries = session.query(BrainTumorTest).all()
        
        # Optionally, convert the result to a list of dictionaries or another format
        # that's easier to work with or serialize (e.g., for JSON responses in a web application)
        result = [
            {
                "id": entry.id,
                "patient_name": entry.patient_name,
                # "image": entry.image,  # Be careful with binary data if serializing
                "predicted_tumor": entry.predicted_tumor,
                "created_date": entry.created_date
            } for entry in all_entries
        ]
        
        return result
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
