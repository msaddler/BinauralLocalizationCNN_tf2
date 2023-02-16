singularity exec --nv \
-B /home \
-B /cm \
-B /mindhive \
-B /nese \
-B /scratch2 \
-B /om2 \
-B /om4 \
-B /net/vast-storage/scratch/ \
-B /net/weka-nfs.ib.cluster/scratch \
/net/vast-storage/scratch/user/msaddler/vagrant/tensorflow-2.10.0.simg \
./jupyter_notebook.sh $1
