import os

training_dir = 'fruits-360/Test'
# Names are common to both Training and Test
class_names = os.listdir(training_dir)

print(class_names)

