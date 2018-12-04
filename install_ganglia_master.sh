#!/usr/bin/env bash
sudo apt-get install ganglia-monitor rrdtool gmetad ganglia-webfrontend -y
sudo cp /etc/ganglia-webfrontend/apache.conf /etc/apache2/sites-enabled/ganglia.conf
sudo cp ./gmetad.conf /etc/ganglia/gmetad.conf
sudo cp ./gmond_master.conf /etc/ganglia/gmond.conf
sudo /etc/init.d/ganglia-monitor start
sudo /etc/init.d/gmetad start
sudo /etc/init.d/apache2 restart