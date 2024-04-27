# Download and run with Mixtral 8x22B Instruct GGUF model
*A brief document on how to run privateGPT with Mixtral 8x22B Instruct GGUF model*

**Last Update:** April 27, 2024

_tuan t. pham_


* docker build image
* git clone and compile llama.cpp
* Download the model
* Merge the files
* Spin up the docker container


```bash
### docker build image
cd ~/src/privateGPT
time docker-compose -f docker-compose-gpu-8x22b.yaml build

### clone and compile llama.cpp
cd ~/src
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j0
# compiled gguf-split is here

### Download model
cd ~/src
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF
cd Mixtral-8x22B-Instruct-v0.1-GGUF
huggingface-cli download --resume-download MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF *.Q6-*.gguf  --local-dir . --local-dir-use-symlink False

### Merge the files
/path/to/gguf-split --merge Mixtral-8x22B-Instruct-v0.1.Q6-00001-of-00004.gguf Mixtral-8x22B-Instruct-v0.1.Q6.gguf
#~ 2.1 GB / layer for 6b quantization

### Spin up the docker container
cd ~/src/privateGPT
cd models
# Use hardlink instead of softlink so that docker overlay filesystem work with volume mount
ln ../../Mixtral-8x22B-Instruct-v0.1-GGUF/Mixtral-8x22B-Instruct-v0.1.Q6.gguf .
docker-compose -f docker-compose-gpu-8x22b.yaml up -d
# watch logs
docker-compose logs -f privategpt
```


* privateGPT is available on your http://localhost:8001/
* If you do not have `huggingface-cli` available, `pip install huggingface-cli` in your virtualenv.
