#!/bin/bash
echo "sourcing qemu env script."
source /tools/Xilinx/Vitis/2024.1/data/emulation/qemu/comp/qemu/environment-setup-x86_64-petalinux-linux
unset LD_LIBRARY_PATH
echo "qemu settings done."
/tools/Xilinx/Vitis/2024.1/data/emulation/qemu/comp/qemu/sysroots/x86_64-petalinux-linux/usr/bin/qemu-img resize -f raw /home/z.ma/aoe-aie/Versal-Linux-cmd/sw/sd_card.img 4294967296
