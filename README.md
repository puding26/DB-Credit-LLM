# DB-Credit-LLM
An Efficient Credit Scoring LLM


# Experimental Setup
## If you want to reproduce this LLM, use one A800 GPU with PyTorch 2.1.2, Python 3.10, and CUDA 11.8.

# Commands

- ## Set up environment.

  ```
  pip install -r requirements.txt
  
  ```
- ## To train, run the following commands.
  
  ```
  python Third_Train.py --dataset <dataset name>

  ```
- ## To valid, run the following commands.

  ```
  python Fourth_Test.py --dataset <dataset name>

  ```
- ## If you want to curate the datasets from the beginning, please refer to the first and second steps.

- ## The flops calculation is provided in FLOPs_calculation.py

- ## Other details are provided in the paper.


# Acknowledgment

## 1. [DeepSeek](https://arxiv.org/abs/2412.19437)


  
  
  
  
