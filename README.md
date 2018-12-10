# vldb19_tensorflow_nn_experiments

Benchmark for forward-NN on tensorflow

##Mount HVE Disk

To run the CPU version on r5d.x2large (AWS is evil), you have to manully mount the ephemeral0 volume:

lsblk

mkdir my_space

sudo mkfs.ext3 /dev/nvme1n1

sudo mount /dev/nvme1n1 ~/my_space

cd my_space/

sudo chmod 777 -R .

git clone https://github.com/BinhangYuan/vldb19_tensorflow_nn_experiments.git

source activate tensorflow_p36 (For Deep learning AMI)

or 

sudo mkfs.ext3 /dev/nvme0n1

sudo mount /dev/nvme0n1 ~

sudo apt-get update

sudo apt install python-pip -y

sudo pip install tensorflow==1.0.0

for gpu benchmark, install gpu version of tensorflow. 

sudo pip install tensorflow-gpu==1.0.0

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:/lib/nccl/cuda-8" (For Deep learning base AMI)

##Set up Ganglia

####Install the master on one node:

sudo apt-get install ganglia-monitor rrdtool gmetad ganglia-webfrontend

sudo cp /etc/ganglia-webfrontend/apache.conf /etc/apache2/sites-enabled/ganglia.conf

Edit /etc/ganglia/gmetad.conf:

---
data_source "my cluster" localhost 

=> 

data_source "tf cluster" 50 master_ip:8649

---

Edit /etc/ganglia/gmond.conf:

---
cluster {
    name = "unspecified"
    owner = "unspecified"
    latlong = "unspecified"
    url = "unspecified"
}

=>

cluster {
    name = "tf cluster"
    owner = "unspecified"
    latlong = "unspecified"
    url = "unspecified"
}


udp_send_channel {
    mcast_join = 239.2.11.71
    port = 8649
    ttl = 1
}

=>

udp_send_channel {
    host = master_ip
    port = 8649
    ttl = 1
}

udp_recv_channel {
    mcast_join = 239.2.11.71
    port = 8649
    bind = 239.2.11.71
}

=>

udp_recv_channel {
    port = 8649
}

---
Start service:

sudo /etc/init.d/ganglia-monitor start

sudo /etc/init.d/gmetad start

sudo /etc/init.d/apache2 restart

####Install the client on the other nodes

sudo apt-get install ganglia-monitor

Edit /etc/ganglia/gmond.conf:

---
cluster {
    name = "unspecified"
    owner = "unspecified"
    latlong = "unspecified"
    url = "unspecified"
}

=>

cluster {
    name = "tf cluster"
    owner = "unspecified"
    latlong = "unspecified"
    url = "unspecified"
}


udp_send_channel {
    mcast_join = 239.2.11.71
    port = 8649
    ttl = 1
}

=>

udp_send_channel {
    host = master_ip
    port = 8649
    ttl = 1
}

udp_recv_channel {
    mcast_join = 239.2.11.71
    port = 8649
    bind = 239.2.11.71
}

=>

[just remove it]

---

sudo /etc/init.d/ganglia-monitor restart