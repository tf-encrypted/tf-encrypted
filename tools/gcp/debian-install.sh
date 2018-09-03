#!/bin/sh

echo 'export LANGUAGE=en_US.UTF-8' >> .bashrc
echo 'export LANG=en_US.UTF-8' >> .bashrc
echo 'export LANG=en_US.UTF-8' >> .bashrc

# install OS packages
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip 
sudo apt-get install -y git
sudo apt-get install -y screen # for running a tf server
sudo apt-get install -y psmisc # for killall
sudo apt-get install -y tcptrack iftop # for network monitoring

# install tensorflow-encrypted
git clone https://github.com/mortendahl/tf-encrypted.git
cd tf-encrypted
pip3 install -e .
