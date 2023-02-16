#!/bin/bash

user=$(whoami)
node=$(hostname -s)
port=8998

if [ "$1" != "" ]; then
    echo "USING SPECIFIED PORT: $1"
    port=$1
fi

if [ $HOSTNAME == ranvier ] &&  [ -z "$SLURM_CLUSTER_NAME" ]; then
    cluster="ranvier.mit.edu"
else
    cluster="openmind7.mit.edu"
fi

if [ "$user" != "msaddler" ] && [ "$port" == "8998" ]; then
    echo "PORT 8998 IS RESERVED FOR msaddler; PLEASE SPECIFY ANOTHER PORT"
else
    # Print instructions to create ssh tunnel
    echo -e "
    >>> Command to create ssh tunnel: <<<
    ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}
    >>> Address for jupyter-notebook: <<<
    localhost:${port}
    "

    unset XDG_RUNTIME_DIR
    jupyter notebook --no-browser --port=${port} --ip="0.0.0.0"
fi
