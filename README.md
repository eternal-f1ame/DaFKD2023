# DA-FKD: Domain Aware Federated Knowledge Distillation

## Advance Machine Learning Paper - 2

### Aaditya Baranwal (B20EE001)

#### System Requirements

* OS: Ubuntu 20.04/22.04
* Python: >=3.6
* conda: >=4.10.1
* Follow the steps below to install the required packages

```bash
conda create -n <ENV-NAME> python=3.9
conda activate <ENV-NAME>
pip install -r requirements.txt
```

#### Clone the repository

* Navigate to the directory where you want to clone the repository: < DIR >

```bash
git clone <REPO URL>
```

* Navigate to the cloned repository

```bash
cd <DIR>/DA-FKD
```

#### Download the Dataset

* URL:

    ```url
    https://github.com/zalandoresearch/fashion-mnist/tree/master/data
    ```

#### Train the Teacher Model

* Do this **once** to make th sh file executable

    ```bash
    chmod +x run_fed_standalone.sh
    ```

* Run the following command to train the teacher model

    ```bash
    sh run_fed_standalone.sh
    ```
