#!/bin/bash

# store them to /home/petalinux/prose_ins/
rsync --progress -rzah ./txt_ckpts/* kria:/home/petalinux/prose_ins/
