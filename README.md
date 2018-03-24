# Parallel-K_MEANS
Parallel and Distributed version of K_Means algorithm for clustering array N of points(x,y) to K clusters. 2018
the lecturer was boris moruz. afeka college.

this solution contains 64bit Cuda project (version 8.0), Mpi and OpenMp methods.
to run this project you need you will need an Nvidia GPU that compatible with cuda toolkit.
So if you didn't already, install Cuda (possible to install a higher than 8.0, you will just need to open a new cuda project yourself).
Also, you will need to install mpich, which is more difficult to install. 
I downloaded the mpich from here http://www.mpich.org/static/downloads/1.4.1p1/
and then used this guide http://swash.sourceforge.net/online_doc/swashimp/node9.html
After you installed both cuda and mpich you may open my project or create a new one (nvidia cuda project) and copy my files to it.
If it is a new project you will need to add some .c file to it (my .c files or just a temporary file that can be deleted in a minute,
so your project will have c/c++ properties either).
In your solution explorer, right click on your project and open properties, 
please make sure that both cuda and general settings are set to 64bit.
Now, go to C/C++ (still in project properties) and general. in the first line - Additional Include Directories,
you will have to add your mpich include folder, usually will be at C:\Program Files\MPICH2\include 
or else wherever you installed it.
Next go to Linker/General and to Additional Library Directories, 
choose your mpich lib folder, again, usually at C:\Program Files\MPICH2\lib.
Almost done, still in properties, go to Linker/Input, Additional Dependencies and write "mpi.lib;" (without the "").
Now you can add all my files to your solution (if you didn't already) and delete the temporary .c file (if you created new project).
Before you build your solution, open Methods file and go to readFromFile method.
Change the PATH in the fopen call, pay attention that every directory is separated by double "/".
Same action in writeToFile method.
Build your solution.
Open wmpichexec, browse to your solution x64/debug/.exe file and set number of processes to 3 or above. 
I included a sample for input file.
Notice that an output file will appear in the solution folder, and beside that, you will see the results on the mpich console.
The initial results are "First occurrence at t = 0.000000 with q = 0.387547".

good luck.
