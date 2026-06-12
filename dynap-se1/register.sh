#!/bin/bash

OS=$(uname)


###########################################################_MAC
if [[ "$OS" == "Darwin" ]]; then
    echo "running on mac"
    VAR=$(ioreg -p IOUSB | grep -i "Dynapse")

    echo "only $(ioreg -p IOUSB -w0 | grep -E "+-o|USB Product"
) internals \n $(system_profiler SPUSBDataType) "
    if [ -n "$VAR" ]; then
        echo "device found on $OS find at $VAR"
    fi


fi

TARGET_SERIAL="00000007"

JAMES=$(python -c "
import samna
devices = samna.device.get_all_devices()
for d in devices:
    if d.serial_number == '$TARGET_SERIAL':
        print(f'Bus {d.usb_bus_number:03d} Device {d.usb_device_address:03d}')
")

OTHER_BOARDS=$(python -c "import samna; devices = samna.device.get_all_devices(); [print(d) for d in devices]"| while IFS= read -r line; do
    echo -e "  → $line " 
    done)

echo "ok === "


#
if [[ "$OS" == "Linux" ]]; then
    echo "$OS"
    echo "devices avail:"
    python -c "import samna; devices = samna.device.get_all_devices(); [print(d) for d in devices]" | while IFS= read -r line; do
    echo -e "  → $line "
    done

    echo -e "\n"
    # JAMES=$(lsusb  | grep "Thesycon")
    if [[ -n "$JAMES" ]]; then
        echo "007 found "
        echo -e "James: \n$JAMES"
    else
    echo "James (serial $TARGET_SERIAL) not found $OTHER_BOARDS"
    #lsusb | while read line; do echo "$line"; done

    fi
fi
###########################################################_WINDOWS

if [[ "$(hostname)" == "DESKTOP-PC" ]]; then
    hostname
    echo "WINDOWS oje"
fi


echo "IF OSError: <<[Errno 13] Permission denied:>> occur, start venv of samna "
