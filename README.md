# Image-Classifier-Application
Predict the type of flower in a picture using a saved pretrained deep learning model loaded from a checkpoint. Display the predicted top K-values and corresponding probabilities.

## Usage instructions
Run predict.py in the terminal and enter the filepath of a picture of a flower on the local computer in the function call to have the program predict the type of flower and output its name along with the probability of the prediction. The function call need only include the file name and the picture directory, i.e.:
python predict.py --dir 'flowers/valid/12/image_03997.jpg'

Three additional parameters are also available to be defined by the user:

The "--top_k" parameter can be used to display a specific number of the most likely predicted types of flowers (e.g. "--top_k 5"). If this parameter is not defined only the highest probability prediction will be displayed.

The "--gpu" parameter can be set to True to use Nvidia Cuda technology for faster performance. This requires that the user has the necessary graphics hardware and software installed. This parameter is set to FALSE by default since the amount of computing power required for prediction is minimal and the advantage of utilizing the GPU is hence negligible. The GPU is mostly of advantage when performing model training which is considerably more compute-intensive.

The "--checkpoint" parameter can be used to set a specific (previously trained) deep learning model to use for the prediction. By not specifying a checkpoint of a specific deep learning model in the function call, a default checkpoint (checkpoint.py) of a densenet-model with 512 hidden units trained for 3 epochs using a learning rate of 0.001 is utilized for the prediction.
