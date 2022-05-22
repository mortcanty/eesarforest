# Dockerfile with Jupyter widget interface for 
# SAR forest mapping
 
FROM     debian:stretch

MAINTAINER Mort Canty "mort.canty@gmail.com"

ENV     REFRESHED_AT 2021-05-06
 
RUN apt-get update && apt-get install -y \
    python3 \
    build-essential \
    pandoc \
    python3-dev \
    python3-pygments \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    python3-pip   

# jupyter notebook with leaflet
RUN     pip3 install --upgrade pip
RUN     pip3 install numpy
RUN     pip3 install scipy
RUN     pip3 install jupyter
RUN     pip3 install ipyleaflet
RUN     jupyter nbextension enable --py --sys-prefix ipyleaflet

RUN     pip3 install -U cython

RUN     pip3 install matplotlib==3.0.3

EXPOSE 8888

# setup for earthengine
RUN     pip3 install pyasn1 --upgrade
RUN     pip3 install --upgrade setuptools && \
        pip3 install google-api-python-client && \
        pip3 install --upgrade oauth2client && \
        pip3 install pyCrypto && \
        apt-get install -y libssl-dev
RUN     pip3 install earthengine-api

#RUN     pip3 install --upgrade --no-cache-dir setuptools==57.0.0

#RUN     pip3 install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==2.1.3

RUN     jupyter nbextension enable --py --sys-prefix ipyleaflet

# install forest,py
copy    setup.py  /home/setup.py
ADD     eesar /home/eesar
WORKDIR /home
RUN     python3 setup.py install  

COPY    forest_cover.ipynb /home/forest_cover.ipynb

RUN     pip3 install geopy


# ipython notebook startup script
COPY    notebook.sh /
RUN     chmod u+x /notebook.sh

WORKDIR /home  
CMD     ["/notebook.sh"]
