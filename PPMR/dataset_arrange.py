patients_ids = [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 28, 29, 30, 31, 34] # no 33

print("patients_ids num:", len(patients_ids))

import random

random.seed(1234)
random.shuffle(patients_ids)

print("shuffled_patients_ids:", patients_ids)

# patients_ids num: 23
# shuffled_patients_ids: [9, 22, 12, 13, 23, 34, 6, 10, 18, 14, 16, 30, 20, 8, 17, 31, 29, 3, 28, 4, 2, 5, 19]
