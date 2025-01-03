#!/usr/bin/env sh
HOME=`pwd`

cd $HOME/extensions/chamfer_dist
python setup.py install --user

cd $HOME/extensions/emd
python setup.py install --user


