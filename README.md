# WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION FOR FEDERATED LEARNING
This repository implements all experiments in the paper the *WASSERSTEIN DISTRIBUTIONALLY ROBUST OPTIMIZATION FOR FEDERATED LEARNING*.
  
Authors: 

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib, tqdm, pandas, h5py, comet_ml
- To download the dependencies: **pip3 install -r requirements.txt**
- The code can be run on any pc, doesn't require GPU.
  
# Datasets:

- Human Activity Recognition (30 clients)
- Vehicle Sensor (23 clients)
- MNIST (100 clients)
- CIFAR (20 clients): This dataset will be downloaded and generated automatically when runing algorithms.
- Five-digit(5 clients in total): 4 for source domain, one for target domain.

Download Link: 

All dataset after downloading must be stored at folder \data


# Training Adversarial
    - For Mnist dataset: Before running experiment, need to run generate_niid_100users.py to generate MNIST dataset

    <pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 128 --learning_rate 0.001 --gamma 0.1 --num_global_iters 200 --local_epochs 2 --algorithm FedRob --subusers 0.1 --numusers 100 --times 1
    python3 main.py --dataset Mnist --model mclr --batch_size 128 --learning_rate 0.001 --robust 0 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1 
    python3 main.py --dataset Mnist --model mclr --batch_size 128 --learning_rate 0.001 --robust 1 --num_global_iters 200 --local_epochs 2 --algorithm FedAvg --subusers 0.1 --numusers 100 --times 1 

    </code></pre>


## Table comparison for 

                              | Dataset        | Algorithm |         Test accuracy        |
                              |----------------|-----------|---------------|--------------|
                              |                            | Convex        | Non Convex   |
                              |----------------|-----------|---------------|--------------|
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |                |           |               |              |
                              |----------------|-----------|---------------|--------------|
