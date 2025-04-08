import os

training_dir = 'fruits-360/custom_images'
# Names are common to both Training and Test
class_names = os.listdir(training_dir)

print(class_names)
print("Number of classes:")
print(len(class_names))