# Are you in Safe Building?

## Introduction
- Disaster risk management is a global priority, with earthquakes being a significant cause of casualties and economic losses over the past several decades. The impact of an earthquake on a region depends on various factors, including
 the structural integrity of buildings, which is critical in reducing the likelihood of collapse. One key aspect of evaluating seismic vulnerability is assessing the ability of buildings to sustain earthquake loads. Multi-story structures
 with abrupt changes in story stiffness are particularly prone to collapse, making it essential to rapidly and accurately identify potential structural deficiencies across large building inventories.
- Recent advances in data availability have opened new opportunities for improving seismic risk assessment. Street viewimages,such as those from Google Street View (GSV), provide valuable visual informationt hat can be leveraged to identify building types and materials. 
  Such information is essential for developing accurate earthquake exposure models, which help estimate the vulnerability of buildings in a given area. By automating the analysis of theseimages, machine learning (ML) models can provide a faster and more scalable solution for identifying structural characteristics relevant to earthquake preparedness.
  
## Data
- The train data folder consists of five subfolders, each representing a different building class: A, B, C, D and S. Each subfolder contains images of buildings that corresponds to a specific type or material.
- The building classes, represented by letters, cnverted into numerical values as follows: A->1, B->2, C->3, D->4, S->5. The entire train data set contains of 2516 images distributed across these five classes.
- The test folder contains 478 images spanning the same five building classes. In this folder, the images are named subsequently using Image IDs, but the class information is not provided.

  #### Task : Predict the correct class for each of these images


