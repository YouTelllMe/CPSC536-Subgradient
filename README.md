This repository contains implementations of subgradient methods as part of my CPSC536M course project. [https://www.overleaf.com/read/mvcrmbysphmd#e907a1](https://www.overleaf.com/read/mvcrmbysphmd#e907a1). 

### Build 
Initialize a virtual environment with an appropriate Python version. This project was ran with Python 3.10.10 and the following instructions is for MacOS. Thus, it's recommended that a Python version > 3.10.10 is used. In the root 
directory, run
```
python3.10 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
but replace "python3.10" with your Python version. Next, simply go to the methods directory and run the main script. 
```
cd src/methods && python main.py
```
This should allow you to recreate the plot from the report linked above. The problem.py file specifies specific configuration of the task assignment problem. The current setup uses matrices generated from the files "A_4x100.txt" and "P_4x100.txt". We follow notations and techniques from https://www.mit.edu/~dimitrib/Incr_2001.pdf. 
