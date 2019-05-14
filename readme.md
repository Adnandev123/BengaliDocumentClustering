Bengali Document Clustering using Word mover's distance
======================================================

This is part of a bigger project. In this part, it creates clusters of bengali news documents, where the goal is to cluster news of the same story from different newspapers. The module is written in Python, which creates a distance matrix of multiple documents using word mover's distance, create clusters using hierarchical clustering algorithm, automatically detacs the number of clusters and outputs the culsers ina json file. Te program also use multithreading to generate distance matrics faster. 

Also, it can be used as an API, deployed in cherrypy server. Moreover, a visualization of the created clusters can also be shown.

### Application

A realtime searvice can be found created using this methods in <https://news.pipilika.com/>

### Methods

Details of the methods can be found in these papers.

Title: Bengali Document Clustering Using Word Movers Distance. Link: <https://ieeexplore.ieee.org/document/8554598>
Title: From Word Embeddings To Document Distances. Link: <https://proceedings.mlr.press/v37/kusnerb15.pdf>

