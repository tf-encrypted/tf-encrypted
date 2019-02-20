#!/bin/sh

echo 'export LANGUAGE=en_US.UTF-8' >> .bashrc
echo 'export LANG=en_US.UTF-8' >> .bashrc
echo 'export LANG=en_US.UTF-8' >> .bashrc
echo 'export LC_ALL=C' >> .bashrc

# install OS packages
sudo apt update
sudo apt -y upgrade
sudo apt install -y python3-pip
sudo apt install -y git
sudo apt install -y screen # for running a tf server
sudo apt install -y psmisc # for killall
sudo apt install -y tcptrack iftop # for network monitoring

# install tensorflow-encrypted
git clone https://github.com/mortendahl/tf-encrypted.git
cd tf-encrypted
pip3 install -e .
