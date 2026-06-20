# Survey Visualization

Install [MikTex](https://miktex.org/download)

```bash
# export DISTRO="resolute" # Ubuntu 26.04 LTS
export DISTRO="noble" # Ubuntu 24.04 LTS
# export DISTRO="jammy" # Ubuntu 22.04 LTS

curl -fsSL https://miktex.org/download/key | sudo gpg --dearmor -o /usr/share/keyrings/miktex.gpg
echo "deb [signed-by=/usr/share/keyrings/miktex.gpg] https://miktex.org/download/ubuntu ${DISTRO} universe" | sudo tee /etc/apt/sources.list.d/miktex.list
sudo apt-get update && sudo apt-get install miktex
sudo miktexsetup --shared=yes finish
sudo initexmf --admin --set-config-value "[MPM]AutoInstall=1"
```

Install Python dependencies:

```python
uv pip install matplotlib numpy scipy ipykernel
```
