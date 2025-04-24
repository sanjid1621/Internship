docker pull xilinx/vitis-ai:latest
docker run -it --rm -v "C:\intern":/workspace -p 8888:8888 xilinx/vitis-ai:latest
conda activate vitis-ai-pytorch


cd /workspace


ls

# 1. Install extra requirements
pip install torchvision scipy matplotlib

# 2. Run quantization script (save as quantize.py in your mounted folder)
python quantize.py

# 3. Test deployment (save as deploy_test.py)
python deploy_test.py


exit

