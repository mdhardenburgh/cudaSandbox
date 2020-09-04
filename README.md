# cudaSandbox
My public sandbox repo for learning CUDA

## Full CUDA Toolkit Documentation
Follow [this](https://docs.nvidia.com/cuda/index.html) link for the full 
documentation. It includes install guides, relsease notes, programming guides 
and more
 
## CUDA C/C++ x86 Ubuntu Install
The main website for the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

The current version of CUDA at the time of writing this is version 11.0 Update 1. 
To install CUDA for C/C++ go to Download Now. Under target platform click on
Linux then x86_64 then Ubuntu then select your version of Ubuntu and then choose 
your installer type. Follow the instrunctions under base installer. After
installing reboot your computer to load drivers and perform post install actions.

Note: I had issues with apt, so I used aptitude.

Steps for deb local install on Ubuntu 18.04
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.3-450.51.06-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
Next, set your PATH `export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}`

Runfile install only also execute `export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64`
and `${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

To install the samples that came with the CUDA toolkit execute `cuda-install-samples-11.0.sh ~`
in the /usr/local/cuda-11.0/bin/ directory

[Additionall samples in the Nvida CUDA samples github repo](https://github.com/nvidia/cuda-samples)

For the [quick linux inatall guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux).
For the [full linux install guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Install CUDA Cross Compiler for Jetson ARM based Devices
This was obnoxiously difficult and not explained well. [Under section 5 of the install guide documentation for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#cross-platform) it says to "Install repository meta-data package with: $ sudo dpkg -i cuda-repo-cross-<identifier>_all.deb".
I have scoured the internet and could not find any such package with reaonable 
effort. In my opinion, ignore this section for installing the cross compiler.

The easiest way, in my experience, to install the cross compiler is to download 
and install the [NVidia Jetson SDK Manager](https://developer.nvidia.com/nvsdk-manager).
Download the package to your host Ubuntu system and run it. It will install the
host cross compiler stuff and there is an option to install all the Jetson 
Jetpack stuff to your Jetson device too.

Lastly, you need to download the arm-gcc compiler to cross compile code from
a host to Jetson. This is provided on the [linux for tegra page](https://developer.nvidia.com/embedded/linux-tegra).
As of writing this the current version used by jetson is is [GCC 7.3.1](https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit).
Unpack it using your favorite extracting tool and take out the gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu
directory.