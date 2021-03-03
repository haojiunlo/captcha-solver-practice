# SIMPLE CAPTCHA-SOLVER-PRACTICE
1. Get Training Data
    ```bash
    python data_process.py
    ```
2. Run train
   ```bash
   # train with ConvBnNet
   python train.py --Net ConvBnNet --num_epochs 10
   ```
3. Run inference
   ```
   python demo.py --Net ConvBnNet --model_path ConvBnNet_torch_model.pt
   ```