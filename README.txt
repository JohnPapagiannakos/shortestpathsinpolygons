~   Computational Geometry <COMP 416> - (2020)  ~
   
* Topic: Given two points inside a polygon, the goal is to find 
  the shortest path lying inside the polygon between the 
  two points. *
    
* Author's Name  : Ioannis Marios Papagiannakos

------------------------------------------------------------------
* Required    :  python 2.7, gdal, Shapely

* Recommended : Use a virtual environment (like "Anaconda").

------------------------------------------------------------------
* Project Structure:

- libraries       --- project_libs.py
                   |- binarySearchTree.py
                   |- graphs.py 
                   '- dcel.py

- main executable --- main.py

------------------------------------------------------------------
* The included ".pdf" files contain the produced shapefiles
  (Layers: LayerOfInterest + MonotonizedLayer + Shortest path) 
  which were opened & exported from qgis. 
