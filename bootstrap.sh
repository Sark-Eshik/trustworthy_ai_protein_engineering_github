#!/bin/bash

apt update -y
apt install -y python3 python3-venv python3-pip git tmux

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

mkdir -p results figures logs

echo "Bootstrap complete."
