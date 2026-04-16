# Setting up Raspberry Pi with DYNAP-SE1 and Samna
# author: mabid
This guide provides instructions for configuring a Raspberry Pi to work with the DYNAP-SE1 neuromorphic chip using the Samna library.

## Hardware Requirements

- Raspberry Pi (tested on Pi 5, should also work on Pi 4). OS tested was Debian/Pi OS.
- DYNAP-SE1 chip
- (preferably the official) power supply for your Pi model. PoE not tested but should work more reliably in theory.
- USB connection between Pi and DYNAP-SE1

## Configuration Steps

### 1. Raspberry Pi Boot Configuration

The key to getting the Pi to work with the DYNAP-SE1 is to increase the USB current limit. The DYNAP-SE1 requires more power than the default USB configuration provides.

#### For Raspberry Pi 4 and earlier:
Edit the `/boot/config.txt` file and add:
```
max_usb_current=1
```

This setting will:
- Set GPIO38 to high
- Turn on a FET (Field Effect Transistor)
- Connect a second 39K resistor in parallel to an existing one on pin 5 of U13 (the AP2553W6 USB power manager)
- Lift the current limit from 0.6A to 1.2A

#### For Raspberry Pi 5:
The Pi 5 has higher current limits by default (up to 1.6A with the official power supply), but this setting is still needed to properly 
connect to the DYNAP-SE1. Using PoE (power over ethernet) with an official PoE fan/hat attachment extends this to >4A.

### 2. Samna Installation

Install Samna using pip:
```bash
pip install samna==0.17
```

### 3. Known Issues and Solutions

#### PyPI Index Mismatch
There is a known issue where the Samna module installs correctly via pip (using PyPI), but upon import, the package attempts to install/reinstall functions from an incorrect GitLab index.

**Problem**: The package tries to use:
```
https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple
```
And fails to retrieve the relevant build for version 0.17. Installing the latest samna works, however it is incompatible with DYNAP-SE1.
**Proposed Solution**: Ensure the package searches for the matching wheel from:
```
https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple/samna
```

#### Verification Steps
After configuration:
1. Reboot the Raspberry Pi
2. Connect the DYNAP-SE1 via USB
3. Test the connection:
   ```python
   import samna
   # test code here (connect to DYNAP-SE1 same as linux demo)
   ```

## Troubleshooting

### Power Issues
- Ensure you're using the official Raspberry Pi power supply
- Verify the `max_usb_current=1` setting is correctly added to `/boot/config.txt`

### Software Issues
- If Samna import fails, check the PyPI index configuration
- Currently, you must use Samna 0.17

## Testing

After setup, verify the configuration works by connecting to the chip and trying basic Samna functionality.

## Notes

- Official power supply is a safe option, PoE https://www.pi-shop.ch/power-over-ethernet-hat-g-for-raspberry-pi-5 supports up to 5V/5A.  

---

