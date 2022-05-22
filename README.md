# Mapping forest extent with SAR change detection

Author: mortcanty

An Omnibus likelihood ratio test is used to identify forest pixels in time series of Sentinel-1 SAR images on GEE.

Pull and start the Docker container with

	docker run -d -p 8888:8888 --name forest_cover mort/forestdocker

Open the Jupyter notebook

	 forest_cover.ipynb

Stop the container with  

	Docker stop forest_cover

Restart with

	Docker start forest_cover
