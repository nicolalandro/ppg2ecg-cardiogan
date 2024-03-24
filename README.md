<p align="center"> 
<img src="./images/ppg2ecg_fast.gif" width=100% /> 
</p>

# CardioGAN: Attentive Generative Adversarial Network with Dual Discriminators for Synthesis of ECG from PPG

Replicated result
![img](codes/test.png)

## How to run
* [link to download weights](https://github.com/pritamqu/ppg2ecg-cardiogan/releases/download/model_weights/cardiogan_ppg2ecg_generator.zip)
* unzip it into weights folder
* now you can test it with docker compose or podman-compose
```
podman-compose run py bash
$ cd codes
$ python test_cardiogan.py
# you can find codes/test.png as plotted results
```

# References 
Here the knowledge you need for that codebase
* docker, docker-compose or podman, podman-compose
* python
* tensorflow, keras
* ...