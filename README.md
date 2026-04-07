To Install Pip in Linux:
Since you lack administrative (sudo) privileges and do not have pip installed, you can bypass system-wide restrictions by installing pip and your packages directly into your user home directory. [1, 2, 3] 
Here is the step-by-step process to get pip and install packages without a password:
## 1. Install pip for your user only [4] 
You can download the official installation script and run it with the --user flag. This does not require a password. [5, 6] 

# Download the installer
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install for your user, adding the 'break-system-packages' flag # to bypass the PEP 668 restriction on newer Linux systems
python3 get-pip.py --user --break-system-packages

## 2. Add the pip location to your PATH [7, 8] 
Once installed, the pip command is usually located in ~/.local/bin. You need to tell your terminal where to find it: [9] 

# Add to your current session
export PATH="$HOME/.local/bin:$PATH"
# Make it permanent by adding it to your shell configuration
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

## 3. Verify and Install Packages
Now you can use pip to install packages without sudo. You must continue using the --user and --break-system-packages flags to avoid permission errors: [10, 11] 

# Check if it works
pip --version
# Install a package (example: requests)
pip install --user --break-system-packages requests

## Alternative: Use Virtual Environments (Safest) [12] 
If your system already has the venv module (most do), you can avoid the break-system-packages risk entirely by creating an isolated environment. This creates a folder you own where you can install anything: [13] 

# Create the environment
python3 -m venv my_project_env
# Activate it
source my_project_env/bin/activate
# Within this environment, pip is usually already available.# You can install packages without --user or --break-system-packages
pip install requests

Which specific Python package are you trying to use? Knowing the package name can help me provide more tailored instructions.

[1] [https://www.activestate.com](https://www.activestate.com/resources/quick-reads/how-to-manually-install-python-packages/)
[2] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[3] [https://stackoverflow.com](https://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges)
[4] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/1agr0mk/how_do_i_download_python_on_computers_without/)
[5] [https://superuser.com](https://superuser.com/questions/1159645/installing-pip-without-root-access)
[6] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[7] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[8] [https://askubuntu.com](https://askubuntu.com/questions/952891/pip-wont-run-without-sudo)
[9] [https://gist.github.com](https://gist.github.com/rhshah/fb634d6e90ebdabedab0dbbeaedbb438)
[10] [https://stackoverflow.com](https://stackoverflow.com/questions/32492188/recommended-way-to-install-a-python-package-with-pip-and-no-sudo-privileges)
[11] [https://askubuntu.com](https://askubuntu.com/questions/363300/how-to-install-pip-python-to-user-without-root-access)
[12] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[13] [https://www.developersoapbox.com](https://www.developersoapbox.com/install-python-packages-without-root-permissions-with-virtualenv/)


Since you lack administrative (sudo) privileges and do not have pip installed, you can bypass system-wide restrictions by installing pip and your packages directly into your user home directory. [1, 2, 3] 
Here is the step-by-step process to get pip and install packages without a password:
## 1. Install pip for your user only [4] 
You can download the official installation script and run it with the --user flag. This does not require a password. [5, 6] 

# Download the installer
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install for your user, adding the 'break-system-packages' flag # to bypass the PEP 668 restriction on newer Linux systems
python3 get-pip.py --user --break-system-packages

## 2. Add the pip location to your PATH [7, 8] 
Once installed, the pip command is usually located in ~/.local/bin. You need to tell your terminal where to find it: [9] 

# Add to your current session
export PATH="$HOME/.local/bin:$PATH"
# Make it permanent by adding it to your shell configuration
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

## 3. Verify and Install Packages
Now you can use pip to install packages without sudo. You must continue using the --user and --break-system-packages flags to avoid permission errors: [10, 11] 

# Check if it works
pip --version
# Install a package (example: requests)
pip install --user --break-system-packages requests

## Alternative: Use Virtual Environments (Safest) [12] 
If your system already has the venv module (most do), you can avoid the break-system-packages risk entirely by creating an isolated environment. This creates a folder you own where you can install anything: [13] 

# Create the environment
python3 -m venv my_project_env
# Activate it
source my_project_env/bin/activate
# Within this environment, pip is usually already available.# You can install packages without --user or --break-system-packages
pip install requests

Which specific Python package are you trying to use? Knowing the package name can help me provide more tailored instructions.

[1] [https://www.activestate.com](https://www.activestate.com/resources/quick-reads/how-to-manually-install-python-packages/)
[2] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[3] [https://stackoverflow.com](https://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges)
[4] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/1agr0mk/how_do_i_download_python_on_computers_without/)
[5] [https://superuser.com](https://superuser.com/questions/1159645/installing-pip-without-root-access)
[6] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[7] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[8] [https://askubuntu.com](https://askubuntu.com/questions/952891/pip-wont-run-without-sudo)
[9] [https://gist.github.com](https://gist.github.com/rhshah/fb634d6e90ebdabedab0dbbeaedbb438)
[10] [https://stackoverflow.com](https://stackoverflow.com/questions/32492188/recommended-way-to-install-a-python-package-with-pip-and-no-sudo-privileges)
[11] [https://askubuntu.com](https://askubuntu.com/questions/363300/how-to-install-pip-python-to-user-without-root-access)
[12] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[13] [https://www.developersoapbox.com](https://www.developersoapbox.com/install-python-packages-without-root-permissions-with-virtualenv/)


Since you lack administrative (sudo) privileges and do not have pip installed, you can bypass system-wide restrictions by installing pip and your packages directly into your user home directory. [1, 2, 3] 
Here is the step-by-step process to get pip and install packages without a password:
## 1. Install pip for your user only [4] 
You can download the official installation script and run it with the --user flag. This does not require a password. [5, 6] 

# Download the installer
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# Install for your user, adding the 'break-system-packages' flag # to bypass the PEP 668 restriction on newer Linux systems
python3 get-pip.py --user --break-system-packages

## 2. Add the pip location to your PATH [7, 8] 
Once installed, the pip command is usually located in ~/.local/bin. You need to tell your terminal where to find it: [9] 

# Add to your current session
export PATH="$HOME/.local/bin:$PATH"
# Make it permanent by adding it to your shell configuration
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

## 3. Verify and Install Packages
Now you can use pip to install packages without sudo. You must continue using the --user and --break-system-packages flags to avoid permission errors: [10, 11] 

# Check if it works
pip --version
# Install a package (example: requests)
pip install --user --break-system-packages requests

## Alternative: Use Virtual Environments (Safest) [12] 
If your system already has the venv module (most do), you can avoid the break-system-packages risk entirely by creating an isolated environment. This creates a folder you own where you can install anything: [13] 

# Create the environment
python3 -m venv my_project_env
# Activate it
source my_project_env/bin/activate
# Within this environment, pip is usually already available.# You can install packages without --user or --break-system-packages
pip install requests

Which specific Python package are you trying to use? Knowing the package name can help me provide more tailored instructions.

[1] [https://www.activestate.com](https://www.activestate.com/resources/quick-reads/how-to-manually-install-python-packages/)
[2] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[3] [https://stackoverflow.com](https://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges)
[4] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/1agr0mk/how_do_i_download_python_on_computers_without/)
[5] [https://superuser.com](https://superuser.com/questions/1159645/installing-pip-without-root-access)
[6] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[7] [https://oit.colorado.edu](https://oit.colorado.edu/services/web-content-applications/python-install-modules)
[8] [https://askubuntu.com](https://askubuntu.com/questions/952891/pip-wont-run-without-sudo)
[9] [https://gist.github.com](https://gist.github.com/rhshah/fb634d6e90ebdabedab0dbbeaedbb438)
[10] [https://stackoverflow.com](https://stackoverflow.com/questions/32492188/recommended-way-to-install-a-python-package-with-pip-and-no-sudo-privileges)
[11] [https://askubuntu.com](https://askubuntu.com/questions/363300/how-to-install-pip-python-to-user-without-root-access)
[12] [https://www.reddit.com](https://www.reddit.com/r/learnpython/comments/5cmr02/installing_pip_modules_without_sudo_should_i/)
[13] [https://www.developersoapbox.com](https://www.developersoapbox.com/install-python-packages-without-root-permissions-with-virtualenv/)


To install all those libraries in your restricted environment (without a password and bypassing system restrictions), run this single command in a terminal:

pip install --user --break-system-packages torch torchvision torchaudio matplotlib seaborn scikit-learn tqdm numpy pillow

## Run it inside your Jupyter Notebook
If you want to run it directly from a notebook cell (which ensures the libraries go to the right place), use this:

%pip install --user --break-system-packages torch torchvision torchaudio matplotlib seaborn scikit-learn tqdm numpy pillow

## Breakdown of what this installs:

* torch, torchvision, torchaudio: The core PyTorch ecosystem.
* matplotlib, seaborn: For your plotting and confusion matrix visuals.
* scikit-learn: Provides confusion_matrix and classification_report.
* tqdm: The progress bar for your training loops.
* numpy: Base numerical library (usually comes with Torch, but good to ensure).
* pillow: The library behind from PIL import Image.

Note: os and io are built into Python, so you don't need to install them.
Important: After the installation finishes, you must restart your Jupyter kernel (Kernel > Restart) before running your import statements again.
Do you have a GPU (NVIDIA) on this machine, or are you planning to run your model on the CPU?

