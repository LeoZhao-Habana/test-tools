#!/bin/bash
HOSTS=10.32.105.80,10.32.105.110
source ./tools/utils.sh
scp_gather $1 $2
