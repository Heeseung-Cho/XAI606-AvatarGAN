# Avatar Image Generator

   Faces Domain <br/>
   <img src="images/Faces_example.jpeg" width="500" />

   Generated Cartoons <br/>
   <img src="images/Cartoons_example.jpeg" width="500" />

   Based on the paper XGAN: https://arxiv.org/abs/1711.05139

## Dataset

  Faces dataset: we use the VggFace dataset (https://www.robots.ox.ac.uk/~vgg/data/vgg_face/) from the University of Oxford

  Cartoon dataset: we use the CartoonSet dataset from Google (https://google.github.io/cartoonset/), both the versions of 10000 and 100000 items.
  
  We filtered out the data just to keep realistic cartoons and faces images. This code is in `scripts`. To download the dataset:
  
  1. `pip3 install gdown`
  2. `gdown https://drive.google.com/uc?id=1tfMW5vZ0aUFnl-fSYpWexoGRKGSQsStL`
  3. `unzip datasets.zip`

## Directory structure

  `config.json`: contains the model configuration to train the model and deploy the app
  
  `weights`: contains weights that we saved the last time we train the model. 

```
├── app.py
├── avatar-image-generator-app
├── config.json
├── Dockerfile
├── images
│   ├── Cartoons_example.jpeg
│   └── Faces_example.jpeg
├── LICENSE
├── losses
│   └── __init__.py
├── models
│   ├── avatar_generator_model.py
│   ├── cdann.py
│   ├── decoder.py
│   ├── denoiser.py
│   ├── discriminator.py
│   ├── encoder.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── scripts
│   ├── copyFiles.sh
│   ├── download_faces.py
│   ├── keepFiles.sh
│   ├── plot_utils.py
│   └── preprocessing_cartoons_data.py
├── sweeps
│   ├── sweep-bs-1.yaml
│   └── sweep-rs-1.yaml
├── train.py
├── utils
│   └── __init__.py
└── weights
    ├── c_dann.pth
    ├── d1.pth
    ├── d2.pth
    ├── denoiser.pth
    ├── disc1.pth
    ├── d_shared.pth
    ├── e1.pth
    ├── e2.pth
    └── e_shared.pth
```
## The model
Our codebase is in Python3. We suggest creating a new virtual environment.
   * The required packages can be installed by running `pip3 install -r requirements.txt`
   * Update `N_CUDA` by running `export N_CUDA=<gpu_number>` if you want to specify the GPU to use 

   It is based on the XGAN paper omitting the Teacher Loss and adding an autoencoder in the end. The latter was trained to learn well only the representation of the cartoons as to "denoise" the spots and wrong colorisation from the face-to-cartoon outputs of the XGAN.

   The model was trained using the hyperparameters located in `config.json`. Weights & Biases Sweep was used to find the best hyperparameters:

1. Change `root_path` in `config.json`. It specifies where is `datasets` which contains the datasets. 
2. Run `wandb login` to use wandb to get the report
3. Run `python3 train.py --wandb --run_name <RUN_NAME>` or `python3 train.py --no-wandb`
4. To launch an agent with a sweep configuration of wandb in bg from ssh `nohup wandb agent --count <RUN_NUMBERS> stevramos/avatar_image_generator/<SWEEP_ID> &`

  You can see the Weights & Biases report here: https://wandb.ai/stevramos/avatar_image_generator
  
  This is the implementation of [our project](https://madewithml.com/projects/1233/generating-avatars-from-real-life-pictures/) created for the Made With ML Data Science Incubator (deprecated).
