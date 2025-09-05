import numpy as np

labels = [
    "sawasdee", "nose_burn", "wound_inflamed", "blurry_vision", "migraine", "cold",
    "tired", "sick", "heartbeat_fast", "arm_pain", "heart_stop", "red_eyes",
    "swollen", "dizzy", "neurosis", "psychosis", "stomach_burn", "allergic",
    "astigmatism", "jaw_pain", "aids", "mentally_disabled", "high_fever",
    "tooth_filling", "epilepsy", "disease"
]

np.save("label_classes.npy", np.array(labels))
print("✅ Save แล้วจ้าจุ้บๆๆ\nLabels:", labels)

      

