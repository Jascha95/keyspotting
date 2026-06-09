#!/bin/bash


echo "Script running"


OS=$(uname)

echo $OS "aka aka OSX(mac)"
if [[ $OS == "Darwin" ]]; then
    echo "running on mac"
    VAR=$(ioreg -p IOUSB | grep -i "Dynapse")
    echo "only $(ioreg -p IOUSB -w0 | grep -E "+-o|USB Product"
) internals \n $(system_profiler SPUSBDataType) "
    if [ -n "$VAR" ]; then
        echo "device found on $OS find at $VAR"
    fi


fi

if [[ $OS != "Darwin" ]]; then
    echo "$OS"
    echo "running on linux"
fi