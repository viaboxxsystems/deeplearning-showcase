# deeplearning-showcase


## Installing Nvidia GPU related requirements

- install Microsoft Visual Studio. (https://visualstudio.microsoft.com/downloads/)
  donwload and install the community edition. When installing choose to install "Visual Studio Build Tools 2017" and "Visual Studio Community 2017" 

- install Cuda Toolkit. (https://developer.nvidia.com/cuda-toolkit)
  Be mindful of where it is installed. The directory has to be added in PATH variable later.
  
- install the drivers for your GPU (https://www.nvidia.com/Download/index.aspx)

- Create an Nvidia developer profile, and download CuDNN (https://developer.nvidia.com/cudnn)
  Download and extract the file in a known directory.

- Setup PATH variable with the following entries
```
<Path to cuda toolkit>\bin
<Path to cuda toolkit>\libnvvp
<Path to cuDNN>\bin
# for example
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
C:\cuda\bin
```  

## Installing python requirements 
install the requirements using the command:
```
pip3 install -r requirements.txt
```


## Setup Keras & Tensorflow with Virtualenv

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

    
## Visual Studio Code and Virtualenv
If you are using Virtual Studio Code you can configure the Editor to use the Virtualenv Python binary. To do this you first need to configure your Virtualenv Root folder with this user setting:

```json
  "python.venvPath": "~/venv",
```  

After a restart you can select your Python Interpreter (Shift+Cmd+P -> "Python: Select Interpreter" -> Select the interpreter in the deeplearning venv, e.g. ~/venv/deeplearn/bin/python). This allows VSCode to access the packages installed in the virtualenv and the editor will use the virtualenv for installing new packages.

## Test Data

At first you will need to verify your kaggle account and accept the competition terms and conditions. Follow this link to do so (https://www.kaggle.com/c/dogs-vs-cats/rules).

Then you can use resources/prepare_data.sh script to configure the directories and download the training and validation data
```
./resources/prepare_data.sh
```
## Tensorboard
If you have tensorboard installed, you can view the visualisations metrics
```bash
tensorboard --logdir=./tensorboard
```

After starting tensorboard you can open your browser on ```localhost:6006``` to view the tensorboard visualization.
