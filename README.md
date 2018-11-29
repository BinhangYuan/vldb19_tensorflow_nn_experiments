# vldb19_tensorflow_nn_experiments

Benchmark for forward-NN on tensorflow

To run the CPU version on r5d.x2large (AWS is evil), you have to manully mount the ephemeral0 volume:

lsblk

mkdir my_space

sudo mkfs.ext3 /dev/nvme1n1

sudo mount /dev/nvme1n1 ~/my_space

cd my_space/

sudo chmod 777 -R .
