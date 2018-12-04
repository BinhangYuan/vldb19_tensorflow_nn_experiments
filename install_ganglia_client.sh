#!/usr/bin/env bash
sudo apt-get install ganglia-monitor -y
sudo cp ./gmond_client.conf /etc/ganglia/gmond.conf
sudo /etc/init.d/ganglia-monitor restart