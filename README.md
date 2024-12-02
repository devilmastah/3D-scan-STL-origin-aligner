# 3D-scan-STL-origin-aligner
Small Python script to align a 3d scan STL file to the origin for further processing in fusion 360. 
I was fed up with having a creality 3d scanner and no option to do this any easy way from the software itself. I know it is possible in fusion but it is quite a lot more work.
At the moment this propably only works with STL's i have not tried anything else.

- This script was build on python 3.10

Following packages were installed from PIP:
- pip install pyvista
- pip install vtk
- pip install numpy


The command line version is just the basic code that opens an STL file in the same directory with a certain name, see bottom of the script

The Origin aligner.py file i used to make this a exe package that i can drag drop the STL file on with:
- pyinstaller --onefile originaligner.py


When the STL file is opened it opens a 3d representation, you can rotate the model with the mouse. 
First step is to find 3 points for the X/Y plane, hover your mouse above the points, then right click or press P on the keyboard. 
After clicking 3 points it will realign the mesh to the XY plane, close the 3d representation, it will save the XY aligned file as alignedXY.stl, then it opens a new window, click two points to rotate the model around the Z axis to align with the X axsis. 
Closing the window will save alignedXYZ. 

Thats it. Maybe when i find time i will improve on it, or feel free to fork it and make use of it :) 
