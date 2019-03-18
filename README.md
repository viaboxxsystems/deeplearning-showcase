# deeplearning-showcase

# To run on a GPU enabled machine 
some special requirements are needed to run on a GPU machine (paperspace P5000). for that we use "requirements_paperspace.txt" file instead of "requirements.txt":

```
pip3 install -r requirements_paperspace.txt
```

Also, we need to revert from Cuda 9.1 to Cuda 9.0. Please refer to "setting_up_paperspace_environment.txt" for step by step explanation.

# Setup Keras & Tensorflow with Virtualenv

Install virtualenv via pip (make sure to use Python3 pip):


Install virtualenv with pip if you don't have it:
```bash
pip3 install virtualenv
```

With virtualenv, create a new environment in the `~/venv/deeplearn` directory, or where you want to store the virtualenvironment for python.
```bash
virtualenv ~/venv/deeplearn
```

Enter the new environment (you might want to make an alias for this - alias=):
```bash
workon deeplearn
```

Use the requirements.txt file to install the required libraries
```bash
pip3 install -r requirements.txt
```
 
Check the default backend in use for keras:
open ~/.keras/keras.json, for this project we are using tensorflow as backend, so the Json should look like :
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

    
# Visual Studio Code and Virtualenv
If you are using Virtual Studio Code you can configure the Editor to use the Virtualenv Python binary. To do this you first need to configure your Virtualenv Root folder with this user setting:

```json
  "python.venvPath": "~/venv",
```  

After a restart you can select your Python Interpreter (Shift+Cmd+P -> "Python: Select Interpreter" -> Select the interpreter in the deeplearning venv, e.g. ~/venv/deeplearn/bin/python). This allows VSCode to access the packages installed in the virtualenv and the editor will use the virtualenv for installing new packages.

# Test Data

use resources/prepare_data.sh script to configure the directories and download the training and validation data
```
./resources/prepare_data.sh
```
# Tensorboard
If you have tensorboard installed, you can view the visualisations metrics
```bash
tensorboard --logdir=./tensorboard
```

After starting tensorboard you can open your browser on ```localhost:6006``` to view the tensorboard visualization.