Bengali Document Clustering using Word mover's distance
======================================================

This is part of a bigger project. In this part, it creates clusters of bengali news documents, where the goal is to cluster news of the same story from different newspapers. The module is written in Python, which creates a distance matrix of multiple documents using word mover's distance, create clusters using hierarchical clustering algorithm, 
automatically detacs the number of clusters and outputs the culsers ina json file. Te program also use multithreading to generate distance matrics faster.
Also, it can be used as an API, deployed in cherrypy server. 

A realtime searvice can be found created using this methods in <https://news.pipilika.com/>

