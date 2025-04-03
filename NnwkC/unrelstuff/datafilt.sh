#!/bin/bash

find  ~/code/python/PythonMultlayerperc/NnwkC/fp -type f -size -1k -delete

find ~/code/python/PythonMultlayerperc/NnwkC/fp -type f ! -iname "*.jpg" ! -iname "*.jpeg" -delete
find  ~/code/python/PythonMultlayerperc/NnwkC/dp -type f -size -1k -delete

find ~/code/python/PythonMultlayerperc/NnwkC/dp -type f ! -iname "*.jpg" ! -iname "*.jpeg" -delete
