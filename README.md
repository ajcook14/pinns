### Setup:

```bash
git clone https://github.com/ajcook14/pinns.git
cd pinns
python3 -m venv vpinns
source vpinns/bin/activate
pip install -r requirements.txt
```

For developers:

```bash
pip install pip-tools
```

Note I am only using the CPU only version of PyTorch here. If you want the GPU version, you need to change the `requirements.in` and re-compile it.