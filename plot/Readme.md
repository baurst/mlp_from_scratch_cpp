# Generating the plot for comparison

This is just a quick script that generates the plot for comparison on the main page.

To generate the plot do

```bash
# run the mlp, writing the logs to /tmp/mlp_log.txt
./build/src/main /path/to/mnist_train.csv /path/to/mnist_test.csv > /tmp/mlp_log.txt

# create venv, install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# to run the tensorflow version
python tf_mlp.py --logfile /tmp/tf_mlp_log

./create_plot.sh /tmp/tf_mlp_log.npz
```
