# 3D Scan STL Origin Aligner
Small Python script to align a STL file that is result of a 3d scan with the origin for further use in other software. 

I was fed up with having two Creality 3d scanners and no option to align the scanned mesh in any of the "cheap" 3d scanner software for further use in other software. I know it is possible in fusion 360 but it is quite a lot more work.
At the moment this propably only works with STL's i have not tried anything else. I also have not found out what the size limit is yet.

- This script was build on python 3.10

Following packages were installed from PIP:
- pip install pyvista
- pip install vtk
- pip install numpy


The command line version is just the basic code that opens an STL file in the same directory with a certain name, see bottom of the script

The Origin aligner.py file is the same but accepts dragging a file on it after making an exe of it with pyinstaller:
- pyinstaller --onefile originaligner.py


When the STL file is opened it opens a 3d representation, you can rotate the model with the mouse. 
First step is to select at least 3 points for the X/Y plane, hover your mouse above the points, then right click or press "p" on the keyboard. 
The amount of selected points will be noted in the console window, you can go back a point with the button "b" on the keyboard. 
After you are satisfied with the point selection you can press space to process the points, it will rotate the model to the XY plane by best fitting to the selected points. 
After this step you will see the rotated model in the window, it will already be saved as alignedXY.stl 
To proceed to fixing the rotation to the X axis close the 3d viewer, it will open up again after closing for the next step. 

The next step is selecting a line that you want to be parrallel to the X axis. This need to be at least 2 points but can be more, an average vector will be used, so more points might be more accurate. 
Key B and Space do the same thing. After pressing space the new stl will be saved as alignedXYZ.stl

Thats it. Maybe when i find time i will improve on it, or feel free to fork it and make use of it :)
Known issue now, sometimes the point you select actually does not appear (the blue dot), big chance its on the backside of your model! i do not know why this happens. Just undo the last point you added and try again. Sometimes changing the angle of the view helps a lot! 

This was quickly thrown together with ChatGPT :)
