def glioma():
    content = """
                Glioma

                Description: Gliomas are a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells, which surround nerve cells and help them function. They can vary greatly in their prognosis, depending on the tumor's exact type, location, and stage.

                Treatments:
                Surgery to remove as much of the tumor as possible.
                Radiation Therapy to destroy tumor cells or slow their growth.
                Chemotherapy to kill tumor cells.
                Targeted Therapy focuses on specific abnormalities in cancer cells.
                Tumor Treating Fields (TTF) Therapy uses electric fields to disrupt cancer cell division.

                Additional Information: Treatment plans depend on the tumor's location, size, type, and the patient's overall health. Ongoing research into gene therapy and immunotherapy offers hope for new treatment options.
                """
    return content

def meningioma():
    content = """
                Meningioma

                Description: Meningiomas are tumors that arise from the meninges, the membranes that envelop the brain and spinal cord. Most meningiomas are benign and grow slowly.

                Treatments:
                Observation for small, asymptomatic tumors.
                Surgical Removal is often possible and is the preferred treatment for symptomatic meningiomas.
                Radiation Therapy for tumors that are difficult to access surgically or for residual tumor cells post-surgery.

                Additional Information: While most meningiomas are benign, they can still cause problems due to their size or location. Regular monitoring through imaging tests may be required for smaller, non-growing tumors.
                """
    return content

def pituitary():
    content = """
                Pituitary Tumors

                Description: Pituitary tumors are abnormal growths that develop in the pituitary gland. Some pituitary tumors result in the overproduction of hormones, while others can cause the gland to produce lower levels of hormones.

                Treatments:
                Medications can often shrink or control the tumor.
                Surgery to remove the tumor, typically done through a minimally invasive technique.
                Radiation Therapy to destroy tumor tissue not removed by surgery.
                
                Additional Information: The pituitary gland plays a critical role in regulating various hormonal functions in the body. Treatment and monitoring by an endocrinologist are often part of managing pituitary tumors.
                """
    return content


def get_description(prediction):
    if prediction.lower() == "glioma":
        description = glioma()
    elif prediction.lower() == "meningioma":
        description = meningioma()
    elif prediction.lower() == "pituitary":
        description = pituitary()
    else:
        description = "The patient has no tumor as identified"
    return description

