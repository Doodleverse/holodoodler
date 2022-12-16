# Doodler

This repository hosts `Doodler`, a Python application that allows interactive construction of sparse labeling for image segmentation using deep neural networks.

## Project background

Deep neural networks have proven to be powerful tools for automatic classification of images (e.g. identifying regions of sand, water and vegetation from drone-collected beach images).  These techniques are used extensively by USGS researchers to understand how beach and coastal environments are impacted by storms and other events. The first step in using this technique is sparsely annotating a collection of images by hand to train the models.

The USGS currently has an open source package - [dash-doodler](https://github.com/dbuscombe-usgs/dash_doodler) - that performs the task, but does not run in multi-user JupyterHub environments. The USGS needs a version that is fully usable from Jupyter Notebooks so that researchers, interns and volunteers can effectively start working immediately in a JupyterHub environment.

The USGS seeks to achieve this goal by using the HoloViz/Panel ecosystem, which will integrate into their existing deployed JupyterHub environment.  

## Installation

Run the following command to create a *conda* environment named `holodoodler`:

```
conda env create --file environment_dev.yaml
```
Make sure to activate the conda environment with:
```
conda activate holodoodler
```
before launching the application with any of the following commands.

## Development

The application currently lives in two places:
* in the `app.py` script: the application can be served by Panel
* in the the Jupyter notebook `doodler_nb.ipynb`: this notebook shows the development process of the app where (1) a series of small and well defined components are imported, (2) then they're put together in an `Application` class that links them when required, (3) a simple layout is built and displayed in the notebook for testing purposes, and (4) a more polished `template` app is built based on a Panel template and is made deployable.

The notebook can be opened with:

```
jupyter lab doodler_nb.ipynb
```

The application can be launched with:

```
panel serve app.py --show
```

Note that the application could also be launched with the following command:

```
panel serve doodler_nb.ipynb
```

## Using Tools
Instructions on how to use the tools are provided at the bottom left of the application.
![Tools_Instructions](https://raw.githubusercontent.com/venuswku/holodoodler/modify-doodles/assets/Tools_Instructions.jpg)
The Tap tool is for selecting doodles, and the Freehand Draw tool is for drawing doodles. All available tools are to the right of the input image.
![All_Tools](https://raw.githubusercontent.com/venuswku/holodoodler/modify-doodles/assets/All_Tools.jpg)
Here is a demo of how to use the tools:
**Tip: If you used the Tap tool to select doodles for removal, make sure to reactivate the Freehand Draw tool afterwards to continue doodling.** Active tools are marked with a blue border on the left side.

## Video (not up to date)

This video shows the notebook content, how to use the application and the state of the application when deployed.


https://user-images.githubusercontent.com/35924738/147021690-5e525c4c-326a-491b-9d27-994250ff9fbb.mp4

