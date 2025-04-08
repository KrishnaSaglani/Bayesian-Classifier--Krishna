# Bayesian-Classifier--Krishna
Fruit Classifier using Bayesian Decision Theory


This Classifier is built on basis of the naive gaussian bayes decision theory. 
I have also incorporated pca feature reduction, which helps
reduce noisy data and reduces my features from 1061 to nearly 440.

To use this classifier, kindly follow the following steps:

1.                  run custom_extractor.py 
What this does:
    This is a feature extractor.
    All the corresponding data generated as a result will 
    be stored in the features folder. 
    This will take a lot of time, so I have done it already.
    So, you can directly run main.py and see the results yourself.


2.                  run main.py
What this does:
    a.First, PCA and data normalisation will be run on the extracted features.

    b.Then training starts.
    The trained model data will thus be stored in "model" folder, along
    with a detailed training log.

    c.Then Classification is called.
    The final results will be generated and stored in "results2" folder.
    The final results shall include
    a. a detailed performance metric analysis of entire model
    b. confusion matrix image for the classification
    c. a log file describing all the processes that were involved 
    in the final classification. 

    d.Finally, custom_classify is called, which is a special function
    that is created specially to classify any fruit you have pictures of. The results of this will be stored in the folder "results_custom".

    Note: To use this feature, please store a all your pics in the folder:
    fruits-360/custom_images. 

3.                  Results

They will be stored in the folder:
results2 (for data of provided dataset ) and 
custom_results (for any custom data that you have entered 
into folder fruits-360/custom_images)
Of course, all results will be shown on terminal when you run 
main.py as well.

                    Thank You.
        We hope you like the 'fruits' of our labour:)
